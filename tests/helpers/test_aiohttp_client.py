"""Tests for the aiohttp client session helper."""

from __future__ import annotations

import asyncio
from base64 import b64encode
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

from music_assistant.helpers import aiohttp_client
from music_assistant.helpers.aiohttp_client import get_proxy_request_kwargs, parse_http_proxy_url


@pytest.fixture
async def proxy_server() -> AsyncGenerator[tuple[str, list[str]]]:
    """Run a local forward proxy that records the URLs it is asked to fetch."""
    requested: list[str] = []

    async def _handler(request: web.Request) -> web.Response:
        requested.append(request.raw_path)
        return web.Response(text="via proxy")

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", _handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    yield f"http://127.0.0.1:{port}", requested
    await runner.cleanup()


async def test_clientsession_honors_proxy_env(
    proxy_server: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test requests are routed through the proxy set in the environment."""
    proxy_url, requested = proxy_server
    monkeypatch.setenv("HTTP_PROXY", proxy_url)
    monkeypatch.setenv("http_proxy", proxy_url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    mass = MagicMock()
    mass.version = "test"
    # the hostname is unresolvable, so the request can only succeed via the proxy
    with patch.object(aiohttp_client, "_get_resolver", return_value=None):
        session = aiohttp_client.create_clientsession(mass)
        async with session.get("http://origin.invalid/hello") as response:
            assert await response.text() == "via proxy"
        await session.close()
    assert requested == ["http://origin.invalid/hello"]


async def test_clientsession_ignores_env_without_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the environment is not trusted when no proxy is configured."""
    monkeypatch.setattr(aiohttp_client, "getproxies", dict)
    mass = MagicMock()
    mass.version = "test"
    with patch.object(aiohttp_client, "_get_resolver", return_value=None):
        session = aiohttp_client.create_clientsession(mass)
        assert session.trust_env is False
        await session.close()


@pytest.mark.parametrize(
    ("entered", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("10.0.0.5:3128", "http://10.0.0.5:3128"),
        (" proxy.lan:8080 ", "http://proxy.lan:8080"),
        ("HTTP://Proxy.LAN:3128", "http://proxy.lan:3128"),
        ("https://proxy.lan:443/", "https://proxy.lan:443"),
        ("http://proxy.lan:80", "http://proxy.lan:80"),
        ("[::1]:8080", "http://[::1]:8080"),
        # the path, query and fragment of a pasted address mean nothing to a proxy
        ("http://proxy.lan:8080/x?y#z", "http://proxy.lan:8080"),
        # credentials are kept as written, percent-encoding included
        ("http://user:p%40ss@proxy.lan:8080", "http://user:p%40ss@proxy.lan:8080"),
    ],
)
def test_parse_http_proxy_url(entered: str, expected: str) -> None:
    """A proxy is normalized to a full URL with an explicit port, and none stays empty."""
    parsed = aiohttp_client.parse_http_proxy_url(entered)
    assert parsed == expected
    # the stored form must survive being parsed again on the next load
    assert aiohttp_client.parse_http_proxy_url(parsed) == parsed


@pytest.mark.parametrize(
    "entered",
    [
        "socks5://proxy.lan:1080",
        "ftp://proxy.lan:21",
        "http://proxy.lan",
        "proxy.lan",
        "http://:8080",
        "proxy.lan:99999",
        "proxy.lan:port",
    ],
)
def test_parse_http_proxy_url_rejects_what_is_not_an_http_proxy(entered: str) -> None:
    """Only http(s) proxies with a host and a port are accepted."""
    with pytest.raises(ValueError, match=r"."):
        aiohttp_client.parse_http_proxy_url(entered)


def test_proxy_request_kwargs_without_a_proxy() -> None:
    """No proxy adds nothing to a request."""
    assert aiohttp_client.get_proxy_request_kwargs("") == {}


def test_proxy_request_kwargs_move_credentials_out_of_the_url() -> None:
    """The credentials of the proxy go in as its auth header, decoded, not in the proxy URL."""
    kwargs = aiohttp_client.get_proxy_request_kwargs("http://user:p%40ss@proxy.lan:8080")
    assert kwargs["proxy"] == "http://proxy.lan:8080"
    assert kwargs["proxy_headers"] == {
        "Proxy-Authorization": aiohttp.encode_basic_auth("user", "p@ss")
    }
    assert "proxy_headers" not in aiohttp_client.get_proxy_request_kwargs("http://proxy.lan:8080")


@pytest.fixture
async def connect_recorder() -> AsyncGenerator[tuple[str, list[str]]]:
    """Run a local proxy that records the head of each request, refusing every tunnel."""
    heads: list[str] = []

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        heads.append((await reader.readuntil(b"\r\n\r\n")).decode())
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield f"127.0.0.1:{port}", heads
    server.close()
    await server.wait_closed()


async def test_proxy_request_kwargs_route_a_request_through_the_proxy(
    proxy_server: tuple[str, list[str]],
) -> None:
    """A plain request made with the kwargs is handed to the proxy."""
    proxy_url, requested = proxy_server
    # the origin is unresolvable, so the request can only succeed via the proxy
    async with (
        aiohttp.ClientSession() as session,
        session.get(
            "http://origin.invalid/hello", **aiohttp_client.get_proxy_request_kwargs(proxy_url)
        ) as response,
    ):
        assert await response.text() == "via proxy"
    assert requested == ["http://origin.invalid/hello"]


async def test_proxy_request_kwargs_log_in_to_the_proxy_for_https(
    connect_recorder: tuple[str, list[str]],
) -> None:
    """The tunnel Spotify's HTTPS traffic travels through is asked for with the proxy login."""
    address, heads = connect_recorder
    configured = parse_http_proxy_url(f"user:p%40ss@{address}")
    async with aiohttp.ClientSession() as session:
        with pytest.raises(aiohttp.ClientHttpProxyError):
            async with session.get(
                "https://origin.invalid/hello", **get_proxy_request_kwargs(configured)
            ):
                pass
    assert len(heads) == 1
    request_line, *headers = heads[0].splitlines()
    assert request_line == "CONNECT origin.invalid:443 HTTP/1.1"
    expected_auth = "Basic " + b64encode(b"user:p@ss").decode()
    assert f"Proxy-Authorization: {expected_auth}" in headers
