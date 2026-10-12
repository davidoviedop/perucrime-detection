"""
Descubrimiento en Facebook: yt-dlp no tiene un extractor útil para el listado
/videos/ de una página, así que los IDs de video se obtienen raspando la
página directamente.

Camino rápido: `requests` simple (sirve para páginas renderizadas en el servidor).
Respaldo: Playwright con Chromium sin interfaz para páginas renderizadas en el
cliente (dependencia opcional: pip install playwright && playwright install chromium).
Ambos aceptan un cookies.txt en formato Netscape para acceso autenticado.
"""
import http.cookiejar
import re

import requests


def _load_cookies_txt(session: requests.Session, cookies_file: str) -> None:
    """Carga un cookies.txt en formato Netscape dentro de una Session de requests."""
    jar = http.cookiejar.MozillaCookieJar(cookies_file)
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
        session.cookies = jar  # type: ignore[assignment]
    except Exception:
        pass


def _parse_netscape_cookies(cookies_file: str) -> list[dict]:
    """Convierte un cookies.txt Netscape en una lista de cookies compatibles con Playwright."""
    cookies: list[dict] = []
    try:
        with open(cookies_file, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue
                domain, _, path, secure, expires, name, value = parts[:7]
                cookies.append({
                    "name":    name,
                    "value":   value,
                    "domain":  domain.lstrip("."),
                    "path":    path,
                    "secure":  secure.upper() == "TRUE",
                    "sameSite": "None",
                })
    except Exception:
        pass
    return cookies


def _fb_entry(video_id: str, page_slug: str) -> dict:
    url = f"https://www.facebook.com/watch/?v={video_id}"
    return {"url": url, "webpage_url": url, "title": video_id, "channel": page_slug}


def _extract_video_ids_from_html(html: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for vid in (
        re.findall(r'"video_id"\s*:\s*"(\d+)"', html)
        + re.findall(r'/videos/(\d{10,20})/?', html)
        + re.findall(r'watch[/?].*?[?&]v=(\d{10,20})', html)
        + re.findall(r'"videoId"\s*:\s*"(\d+)"', html)
    ):
        if vid not in seen and len(vid) >= 10:
            seen.add(vid)
            result.append(vid)
    return result


def _scrape_facebook_page_requests(url: str, cookies_file: str | None) -> str:
    """Devuelve el HTML crudo vía requests; cadena vacía si falla."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "es-PE,es;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    })
    if cookies_file:
        _load_cookies_txt(session, cookies_file)
    try:
        resp = session.get(url, timeout=20, allow_redirects=True)
        return resp.text if resp.status_code == 200 else ""
    except Exception:
        return ""


def _scrape_facebook_page_playwright(url: str, cookies_file: str | None, log) -> list[str]:
    """
    Usa Playwright (Chromium sin interfaz) para renderizar la página de Facebook
    y recolectar IDs de video. Carga las cookies de cookies_file (formato
    Netscape) si se indica.
    Requiere: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        log.warning("    [!] Playwright not installed — run: pip install playwright && playwright install chromium")
        return []

    seen: set[str] = set()
    results: list[str] = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="es-PE",
            )
            if cookies_file:
                pw_cookies = _parse_netscape_cookies(cookies_file)
                if pw_cookies:
                    ctx.add_cookies(pw_cookies)

            page = ctx.new_page()

            # Captura IDs de video de las respuestas XHR (Facebook carga los videos vía GraphQL)
            def on_response(response):
                if "facebook.com" in response.url and response.status == 200:
                    try:
                        text = response.text()
                        for vid in _extract_video_ids_from_html(text):
                            if vid not in seen:
                                seen.add(vid)
                                results.append(vid)
                    except Exception:
                        pass

            page.on("response", on_response)

            try:
                page.goto(url, wait_until="networkidle", timeout=30_000)
            except PWTimeout:
                pass  # una carga parcial es aceptable

            # También extrae del HTML ya renderizado
            html = page.content()
            for vid in _extract_video_ids_from_html(html):
                if vid not in seen:
                    seen.add(vid)
                    results.append(vid)

            # Hace un scroll para disparar la carga diferida de videos
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.5)")
                page.wait_for_timeout(2000)
                html2 = page.content()
                for vid in _extract_video_ids_from_html(html2):
                    if vid not in seen:
                        seen.add(vid)
                        results.append(vid)
            except Exception:
                pass

            browser.close()
    except Exception as exc:
        log.warning(f"    [!] Playwright error: {exc}")

    return results


def scrape_facebook_page(url: str, max_results: int, cookies_file: str | None, log) -> list[dict]:
    """
    Raspa el listado /videos/ de una página de Facebook.
    Camino rápido: analiza el HTML obtenido con requests (páginas SSR).
    Respaldo: Playwright (Chrome sin interfaz) para páginas totalmente renderizadas en el cliente.
    """
    m = re.search(r"facebook\.com/([^/?#]+)", url)
    page_slug = m.group(1) if m else url

    # Camino rápido: basado en requests (sin JS)
    html = _scrape_facebook_page_requests(url, cookies_file)
    video_ids = _extract_video_ids_from_html(html) if html else []

    # Respaldo con Playwright para páginas renderizadas en el cliente
    if not video_ids:
        log.debug(f"    [fb] requests found 0 IDs for '{page_slug}', trying Playwright…")
        video_ids = _scrape_facebook_page_playwright(url, cookies_file, log)

    if not video_ids:
        if not cookies_file:
            log.warning(
                f"    [!] No videos found for '{page_slug}'. "
                "Pass --cookies /path/to/cookies.txt for auth."
            )
        else:
            log.warning(f"    [!] No videos found for '{page_slug}'.")

    return [_fb_entry(vid, page_slug) for vid in video_ids[:max_results]]
