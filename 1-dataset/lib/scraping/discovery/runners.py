"""
Ejecutores de descubrimiento por plataforma. Cada uno recorre las fuentes
configuradas de los países pedidos, convierte cada entrada en una fila de items
y la inserta en el almacén de items (items.csv) con status='discovered'
(omitiendo las URLs que ya están).

Ambos se detienen antes para un país cuando MAX_CONSECUTIVE_ERRORS llamadas
seguidas a yt-dlp lanzan excepción, en vez de repetir el mismo fallo con el
resto de la lista (ver constants.py).
"""
from lib.common.classes import target_class_of
from lib.common.label_rules import assign_label

from .constants import COUNTRY_KEY_MAP, MAX_CONSECUTIVE_ERRORS, PLATFORM_LIST_KEY
from .facebook import scrape_facebook_page
from .items import entry_to_item
from .ytdlp import scrape_page, search_youtube


def _store_entries(entries: list[dict], source: str, country_code: str, store, dry_run: bool) -> tuple[int, int]:
    """Inserta en el almacén (items.csv) las entradas nuevas. Devuelve (insertadas, omitidas_por_duplicadas)."""
    inserted = skipped = 0
    for entry in entries:
        item = entry_to_item(entry, source, country_code)
        if item is None:
            continue
        if dry_run:
            inserted += 1
        elif store.exists(item["id"]):
            skipped += 1
        else:
            store.insert_item(item)
            inserted += 1
    return inserted, skipped


def _country_key(country_code: str) -> str | None:
    return next((k for k, v in COUNTRY_KEY_MAP.items() if v == country_code), None)


def run_youtube(sources_cfg: dict, countries: list[str], max_results: int,
                store, dry_run: bool, log, target_classes: set[str] | None = None) -> tuple[int, int]:
    """YouTube: consultas de búsqueda (ytsearch) + listados de subidas de canales, por país."""
    inserted = skipped = 0
    yt_cfg = sources_cfg.get("youtube", {})

    for country_code in countries:
        country_key = _country_key(country_code)
        if not country_key or country_key not in yt_cfg:
            log.info(f"  [!] No YouTube config for country '{country_code}' — skipping.")
            continue

        country_cfg = yt_cfg[country_key]
        queries  = country_cfg.get("queries",  [])
        channels = country_cfg.get("channels", [])
        log.info(f"\n  [Country: {country_code}]  {len(queries)} queries, {len(channels)} channels")

        consecutive_errors = 0
        for query in queries:
            # El texto de la consulta se clasifica igual que el título de un
            # item: así --classes omite consultas completas desde el inicio, en
            # lugar de descubrir y filtrar todo para luego descartar casi todo
            # en 2_filter.py.
            if target_classes:
                tc = target_class_of(assign_label(query, ""))
                if tc not in target_classes:
                    log.debug(f"    [query] {query}  → skipped (class={tc or 'None'} not in --classes)")
                    continue
            log.info(f"    [query] {query}")
            entries, had_error = search_youtube(query, max_results, log)
            log.info(f"            → {len(entries)} results")
            consecutive_errors = consecutive_errors + 1 if had_error else 0
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error(f"  [!] {consecutive_errors} consecutive yt-dlp errors on youtube search — "
                         "search appears to be down; skipping remaining queries for this country.")
                break
            ins, skip = _store_entries(entries, "youtube", country_code, store, dry_run)
            inserted += ins
            skipped += skip

        consecutive_errors = 0
        for channel_url in channels:
            log.info(f"    [channel] {channel_url}")
            entries, had_error = scrape_page(channel_url, max_results, log)
            log.info(f"              → {len(entries)} results")
            consecutive_errors = consecutive_errors + 1 if had_error else 0
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error(f"  [!] {consecutive_errors} consecutive yt-dlp errors scraping youtube channels — "
                         "the extractor may be down upstream; skipping remaining channels for this country.")
                break
            ins, skip = _store_entries(entries, "youtube", country_code, store, dry_run)
            inserted += ins
            skipped += skip

    return inserted, skipped


def run_page_based(source: str, sources_cfg: dict, countries: list[str],
                   max_per_page: int, store, dry_run: bool, log,
                   fb_cookies: str | None = None) -> tuple[int, int]:
    """Ejecutor genérico para facebook/tiktok/instagram/twitter (raspado de páginas/cuentas)."""
    inserted = skipped = 0
    list_key = PLATFORM_LIST_KEY[source]
    platform_cfg = sources_cfg.get(source, {})

    for country_code in countries:
        country_key = _country_key(country_code)
        if not country_key or country_key not in platform_cfg:
            log.info(f"  [!] No {source} config for country '{country_code}' — skipping.")
            continue

        urls = platform_cfg[country_key].get(list_key, [])
        log.info(f"\n  [Country: {country_code}]  {len(urls)} {list_key}")

        consecutive_errors = 0
        for page_url in urls:
            log.info(f"    [{list_key[:-1]}] {page_url}")
            if source == "facebook":
                # Facebook tiene su propio respaldo requests+Playwright y nunca
                # lanza excepción: allí "0 videos" ya es un resultado final, no
                # una señal de que toda la plataforma esté caída.
                entries = scrape_facebook_page(page_url, max_per_page, fb_cookies, log)
                had_error = False
            else:
                entries, had_error = scrape_page(page_url, max_per_page, log)
            log.info(f"              → {len(entries)} entries")

            consecutive_errors = consecutive_errors + 1 if had_error else 0
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error(f"  [!] {consecutive_errors} consecutive yt-dlp errors scraping {source} — "
                         f"the {source} extractor may be down upstream (yt-dlp sometimes marks a site's "
                         "extractor broken between releases — check `yt-dlp -U` and the yt-dlp issue "
                         f"tracker); skipping remaining {list_key} for this country rather than repeating "
                         "the same failure.")
                break

            ins, skip = _store_entries(entries, source, country_code, store, dry_run)
            inserted += ins
            skipped += skip

    return inserted, skipped
