"""
Tests for routing the Spotify provider's traffic through an HTTP proxy.

The proxy is one optional setting of the instance, collected first by the setup flow and
stored with the rest of its setup data. It has to reach everything the instance does:
the Web API and the token refresh, the sign-in of the setup flow itself, and the
playback of both backends. A direct connection must stay exactly what it was.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from music_assistant_models.errors import SetupFailedError

from music_assistant.helpers.throttle_retry import ThrottlerManager
from music_assistant.models.setup_flow import SetupFlowError
from music_assistant.providers.spotify import helpers as spotify_helpers
from music_assistant.providers.spotify import setup_flow
from music_assistant.providers.spotify.backends.librespot import LibrespotBackend
from music_assistant.providers.spotify.constants import (
    CONF_CLIENT_ID,
    CONF_PROXY_URL,
    CONF_REFRESH_TOKEN_DEV,
    CONF_REFRESH_TOKEN_GLOBAL,
    CREDENTIALS_FILE,
    PAIRING_DEVICE_NAME,
)
from music_assistant.providers.spotify.helpers import (
    get_librespot_proxy_args,
    get_proxy_env,
    get_spotify_token,
    librespot_credentials_via_pairing,
    librespot_credentials_via_token,
)
from music_assistant.providers.spotify.provider import SpotifyProvider
from music_assistant.providers.spotify.session import SpotifySession
from music_assistant.providers.spotify.setup_flow import (
    CONF_PLAYBACK_AUTH_METHOD,
    PLAYBACK_AUTH_APP,
    PLAYBACK_AUTH_BROWSER,
)

PROXY = "http://proxy.lan:3128"
PROXY_WITH_LOGIN = "http://user:p%40ss@proxy.lan:3128"
PROXY_VARIABLES = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
}


def _make_provider(setup_data: dict[str, Any], tmp_path: Path | None = None) -> SpotifyProvider:
    """Return a SpotifyProvider (bypassing __init__) with the given setup_data."""
    prov = object.__new__(SpotifyProvider)
    config = MagicMock(instance_id="spotify--proxy-test")
    config.get_value = MagicMock(return_value=None)
    config.values = {}
    prov.config = config
    prov.manifest = MagicMock(domain="spotify")
    prov.logger = MagicMock()
    prov.available = True
    mass = MagicMock()
    mass.cache_path = str(tmp_path / "cache") if tmp_path else "/nonexistent/cache"
    mass.config.get = MagicMock(return_value=setup_data)
    mass.config.get_raw_provider_config_value = MagicMock(return_value=None)
    mass.config.decrypt_string = MagicMock(side_effect=lambda value: value)
    prov.mass = mass
    return prov


def _response(status: int = 200, payload: Any = None) -> MagicMock:
    """Return a context manager that yields a canned HTTP response."""
    response = MagicMock(status=status, headers={})
    response.json = AsyncMock(return_value=payload)
    response.text = AsyncMock(return_value="")
    return MagicMock(
        __aenter__=AsyncMock(return_value=response), __aexit__=AsyncMock(return_value=None)
    )


# --- the Web API and the token refresh


def _make_session(proxy_url: str) -> tuple[SpotifySession, MagicMock]:
    """Return a Web API session on the given proxy, with the request it makes mocked."""
    mass = MagicMock()
    request = MagicMock(return_value=_response())
    mass.http_session.request = request
    session = SpotifySession(
        mass,
        MagicMock(),
        "global",
        ThrottlerManager(rate_limit=100, period=1),
        get_auth=AsyncMock(return_value={"access_token": "token"}),
        on_unauthorized=MagicMock(),
        proxy_url=proxy_url,
    )
    return session, request


async def test_web_api_requests_go_through_the_proxy() -> None:
    """Every request of a session carries the proxy and its login."""
    session, request = _make_session(PROXY_WITH_LOGIN)
    async with session._request("GET", "me"):
        pass
    kwargs = request.call_args.kwargs
    assert kwargs["proxy"] == PROXY
    # the login is sent as the header of the proxy, never inside the proxy's address
    assert kwargs["proxy_headers"] == {
        "Proxy-Authorization": aiohttp.encode_basic_auth("user", "p@ss")
    }


async def test_web_api_requests_stay_direct_without_a_proxy() -> None:
    """Without a proxy nothing about a request changes."""
    session, request = _make_session("")
    async with session._request("GET", "me"):
        pass
    assert "proxy" not in request.call_args.kwargs
    assert "proxy_headers" not in request.call_args.kwargs


async def test_token_refresh_goes_through_the_proxy() -> None:
    """The refresh of an access token reaches Spotify's accounts service via the proxy."""
    http_session = MagicMock()
    http_session.post = MagicMock(return_value=_response(200, {"expires_in": 3600}))
    await get_spotify_token(http_session, "client", "refresh", "global", PROXY)
    assert http_session.post.call_args.kwargs["proxy"] == PROXY


async def test_token_refresh_stays_direct_without_a_proxy() -> None:
    """A refresh without a proxy is the request it always was."""
    http_session = MagicMock()
    http_session.post = MagicMock(return_value=_response(200, {"expires_in": 3600}))
    await get_spotify_token(http_session, "client", "refresh")
    assert "proxy" not in http_session.post.call_args.kwargs


# --- the logins of the provider


@pytest.mark.parametrize("proxy_url", [PROXY, ""])
async def test_the_logins_refresh_their_tokens_through_the_proxy(
    proxy_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both the shared and the developer login hand the proxy of the instance to Spotify."""
    prov = _make_provider(
        {
            CONF_REFRESH_TOKEN_GLOBAL: "global-token",
            CONF_REFRESH_TOKEN_DEV: "dev-token",
            CONF_CLIENT_ID: "dev-client",
        }
    )
    prov.proxy_url = proxy_url
    prov._auth_info_global = None
    prov._auth_info_dev = None
    prov._sp_user = {"display_name": "tester"}
    refresh = AsyncMock(
        return_value={"access_token": "a", "refresh_token": "r", "expires_at": 9999999999}
    )
    monkeypatch.setattr("music_assistant.providers.spotify.provider.get_spotify_token", refresh)
    monkeypatch.setattr(prov, "_update_setup_data", MagicMock())
    await prov.login()
    await prov.login_dev()
    assert [call.args[4] for call in refresh.await_args_list] == [proxy_url, proxy_url]


# --- loading the provider


async def test_the_provider_reads_its_proxy_once_and_hands_it_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored proxy is normalized at load and used by both Web API sessions."""
    prov = _make_provider({CONF_PROXY_URL: "proxy.lan:3128"}, tmp_path)
    backend = MagicMock(setup=AsyncMock(), unload=AsyncMock())
    monkeypatch.setattr(prov, "_create_backend", MagicMock(return_value=backend))
    monkeypatch.setattr(prov, "login", AsyncMock())
    monkeypatch.setattr(prov, "_test_audiobook_support", AsyncMock(return_value=False))
    monkeypatch.setattr(prov, "_remove_unused_playback_credentials", MagicMock())
    await prov.handle_async_init()
    assert prov.proxy_url == PROXY
    assert prov._global_session._proxy_kwargs == {"proxy": PROXY}
    assert prov._dev_session._proxy_kwargs == {"proxy": PROXY}


async def test_a_provider_without_a_proxy_connects_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No stored proxy, or a cleared one, means no proxy anywhere."""
    for stored in ({}, {CONF_PROXY_URL: None}, {CONF_PROXY_URL: ""}):
        prov = _make_provider(stored, tmp_path)
        backend = MagicMock(setup=AsyncMock(), unload=AsyncMock())
        monkeypatch.setattr(prov, "_create_backend", MagicMock(return_value=backend))
        monkeypatch.setattr(prov, "login", AsyncMock())
        monkeypatch.setattr(prov, "_test_audiobook_support", AsyncMock(return_value=False))
        monkeypatch.setattr(prov, "_remove_unused_playback_credentials", MagicMock())
        await prov.handle_async_init()
        assert prov.proxy_url == ""
        assert prov._global_session._proxy_kwargs == {}


@pytest.mark.parametrize("stored", ["socks5://proxy.lan:1080", "http://proxy.lan", "http://:80"])
async def test_a_proxy_that_cannot_be_used_fails_the_load(stored: str, tmp_path: Path) -> None:
    """A hand-edited proxy that is not an HTTP proxy fails the load with a clear reason."""
    prov = _make_provider({CONF_PROXY_URL: stored}, tmp_path)
    with pytest.raises(SetupFailedError, match="proxy"):
        await prov.handle_async_init()


# --- the playback helpers


def test_proxy_env_names_the_proxy_for_every_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proxy is offered under each of the variable names programs look for."""
    monkeypatch.delenv("NO_PROXY", raising=False)
    env = get_proxy_env(PROXY)
    assert {name for name, value in env.items() if value == PROXY} == PROXY_VARIABLES
    # loopback, where the engine's local API lives, never goes through the proxy
    assert env["NO_PROXY"] == env["no_proxy"] == "localhost,127.0.0.1,::1"


def test_proxy_env_keeps_the_hosts_that_already_bypass_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the server was told to reach directly stays direct."""
    monkeypatch.setenv("NO_PROXY", "ha.lan,.internal")
    assert get_proxy_env(PROXY)["NO_PROXY"] == "ha.lan,.internal,localhost,127.0.0.1,::1"


def test_no_proxy_means_no_environment_and_no_arguments() -> None:
    """Without a proxy the subprocesses are started exactly as before."""
    assert get_proxy_env("") == {}
    assert get_librespot_proxy_args("") == []
    assert get_librespot_proxy_args(PROXY) == ["--proxy", PROXY]


async def test_librespot_authorization_through_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning the token into a playback credential happens through the proxy."""
    calls: list[tuple[tuple[str, ...], dict[str, str] | None]] = []

    async def _fake_check_output(
        *args: str, env: dict[str, str] | None = None, **_kwargs: Any
    ) -> tuple[int, bytes]:
        calls.append((args, env))
        cache_dir = args[args.index("--cache") + 1]
        credentials = json.dumps({"username": "tester", "auth_data": "blob"})
        (Path(cache_dir) / CREDENTIALS_FILE).write_text(credentials, encoding="utf-8")
        return 0, b""

    monkeypatch.setattr(spotify_helpers, "check_output", _fake_check_output)
    await librespot_credentials_via_token("/bin/librespot", "token", PROXY)
    await librespot_credentials_via_token("/bin/librespot", "token")
    (proxied_args, proxied_env), (direct_args, direct_env) = calls
    assert proxied_args[proxied_args.index("--proxy") + 1] == PROXY
    assert proxied_env is not None
    assert proxied_env["HTTPS_PROXY"] == PROXY
    assert "--proxy" not in direct_args
    assert not direct_env


async def test_librespot_pairing_through_the_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pairing daemon reaches Spotify through the proxy once the app selected it."""
    spawns: list[tuple[list[str], dict[str, Any]]] = []

    class _FakeProcess:
        def __init__(self, args: list[str], **kwargs: Any) -> None:
            spawns.append((args, kwargs))

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            return None

        def attach_stderr_reader(self, task: asyncio.Task[None]) -> None:
            task.cancel()

    monkeypatch.setattr(spotify_helpers, "AsyncProcess", _FakeProcess)
    monkeypatch.setattr(spotify_helpers, "_log_pairing_output", AsyncMock())
    monkeypatch.setattr(spotify_helpers, "_await_credentials_file", AsyncMock(return_value="c"))
    await librespot_credentials_via_pairing("/bin/librespot", "Device", None, PROXY)
    await librespot_credentials_via_pairing("/bin/librespot", "Device")
    (proxied_args, proxied_kwargs), (direct_args, direct_kwargs) = spawns
    assert proxied_args[proxied_args.index("--proxy") + 1] == PROXY
    assert proxied_kwargs["env"]["HTTP_PROXY"] == PROXY
    assert "--proxy" not in direct_args
    assert not direct_kwargs["env"]


async def _stream_with(
    proxy_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], dict[str, str]]:
    """Stream one item on librespot and return the argv and environment it was started with."""
    prov = _make_provider({}, tmp_path)
    prov.proxy_url = proxy_url
    prov.cache_dir = str(tmp_path / "cache")
    backend = LibrespotBackend(prov)
    backend._librespot_bin = "/bin/librespot"
    spawned: list[tuple[list[str], dict[str, Any]]] = []

    class _FakeProcess:
        returncode = 0
        proc = None

        def __init__(self, args: list[str], **kwargs: Any) -> None:
            spawned.append((args, kwargs))

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            return None

        def attach_stderr_reader(self, task: asyncio.Task[None]) -> None:
            task.cancel()

        async def iter_chunked(self, _n: int = 64000) -> Any:
            yield b"ogg"

    monkeypatch.setattr(
        "music_assistant.providers.spotify.backends.librespot.AsyncProcess", _FakeProcess
    )
    async for _chunk in backend.stream_spotify_uri("spotify:track:xyz", 0):
        pass
    args, kwargs = spawned[0]
    return args, kwargs["env"]


async def test_librespot_streams_through_the_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each item is fetched through the proxy of the instance."""
    args, env = await _stream_with(PROXY_WITH_LOGIN, tmp_path, monkeypatch)
    assert args[args.index("--proxy") + 1] == PROXY_WITH_LOGIN
    assert env["HTTPS_PROXY"] == PROXY_WITH_LOGIN


async def test_librespot_streams_directly_without_a_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a proxy the process is started as it always was."""
    args, env = await _stream_with("", tmp_path, monkeypatch)
    assert "--proxy" not in args
    assert not env


# --- the setup flow


def _flow_session(*forms: dict[str, Any]) -> MagicMock:
    """Return a setup session that answers its forms with the given values, in order."""
    session = MagicMock()
    session.form = AsyncMock(side_effect=list(forms))
    return session


async def test_an_empty_proxy_asks_for_no_check() -> None:
    """Leaving the proxy empty connects directly and tests nothing."""
    session = _flow_session({CONF_PROXY_URL: ""})
    assert await setup_flow._ask_proxy(session, "") == ""
    session.mass.http_session.get.assert_not_called()


async def test_a_proxy_is_stored_in_its_normalized_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare host and port becomes a full address, once Spotify answers through it."""
    reaches = AsyncMock(return_value=True)
    monkeypatch.setattr(setup_flow, "_proxy_reaches_spotify", reaches)
    session = _flow_session({CONF_PROXY_URL: " proxy.lan:3128 "})
    assert await setup_flow._ask_proxy(session, "") == PROXY
    reaches.assert_awaited_once_with(session, PROXY)
    assert session.form.await_args.kwargs["step_id"] == "proxy"


async def test_the_stored_proxy_is_prefilled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconfiguring shows the proxy that is set."""
    monkeypatch.setattr(setup_flow, "_proxy_reaches_spotify", AsyncMock(return_value=True))
    session = _flow_session({CONF_PROXY_URL: PROXY})
    await setup_flow._ask_proxy(session, PROXY)
    assert session.form.await_args.args[0][0].value == PROXY


async def test_an_unusable_proxy_asks_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A SOCKS or incomplete address is turned back, keeping what was typed."""
    reaches = AsyncMock(return_value=True)
    monkeypatch.setattr(setup_flow, "_proxy_reaches_spotify", reaches)
    session = _flow_session(
        {CONF_PROXY_URL: "socks5://proxy.lan:1080"}, {CONF_PROXY_URL: "proxy.lan:1080"}
    )
    assert await setup_flow._ask_proxy(session, "") == "http://proxy.lan:1080"
    retry = session.form.await_args_list[1]
    assert retry.kwargs["errors"] == {CONF_PROXY_URL: "proxy_invalid"}
    assert retry.args[0][0].value == "socks5://proxy.lan:1080"
    # nothing was tried on the proxy that could not be one
    reaches.assert_awaited_once_with(session, "http://proxy.lan:1080")


async def test_a_proxy_that_does_not_get_through_asks_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy Spotify cannot be reached through is turned back before the sign-in."""
    monkeypatch.setattr(setup_flow, "_proxy_reaches_spotify", AsyncMock(side_effect=[False, True]))
    session = _flow_session({CONF_PROXY_URL: "typo.lan:3128"}, {CONF_PROXY_URL: PROXY})
    assert await setup_flow._ask_proxy(session, "") == PROXY
    retry = session.form.await_args_list[1]
    assert retry.kwargs["errors"] == {CONF_PROXY_URL: "proxy_unreachable"}
    assert retry.args[0][0].value == "typo.lan:3128"


@pytest.mark.parametrize(
    ("status", "reached"),
    [
        (200, True),
        # what Spotify itself answers is of no interest
        (403, True),
        (500, True),
        # but a proxy that wants other credentials is no way through
        (407, False),
    ],
)
async def test_the_proxy_check_looks_only_at_the_proxy(status: int, reached: bool) -> None:
    """Only a missing answer, or the proxy refusing us, fails the check."""
    session = MagicMock()
    session.mass.http_session.get = MagicMock(return_value=_response(status))
    assert await setup_flow._proxy_reaches_spotify(session, PROXY_WITH_LOGIN) is reached
    kwargs = session.mass.http_session.get.call_args.kwargs
    assert kwargs["proxy"] == PROXY
    assert "proxy_headers" in kwargs


@pytest.mark.parametrize("error", [aiohttp.ClientConnectionError("down"), TimeoutError()])
async def test_the_proxy_check_fails_when_nothing_answers(error: Exception) -> None:
    """A proxy that cannot be connected to, or that stays silent, does not get through."""
    session = MagicMock()
    session.mass.http_session.get = MagicMock(side_effect=error)
    assert await setup_flow._proxy_reaches_spotify(session, PROXY) is False


async def test_the_sign_in_goes_through_the_proxy() -> None:
    """The token exchange of the sign-in, and the profile lookup after it, use the proxy."""
    session = MagicMock()
    session.callback_url = "http://ma.lan:8095/setup/callback"
    session.external = AsyncMock(return_value={"code": "abc"})
    token = {"refresh_token": "refresh", "access_token": "access"}
    session.mass.http_session.post = MagicMock(return_value=_response(200, token))
    session.mass.http_session.get = MagicMock(return_value=_response(200, {"id": "u1"}))
    assert await setup_flow._pkce_authenticate(session, "client", "authenticate", PROXY) == token
    assert session.mass.http_session.post.call_args.kwargs["proxy"] == PROXY
    assert await setup_flow._get_profile(session, "access", PROXY) == {"id": "u1"}
    assert session.mass.http_session.get.call_args.kwargs["proxy"] == PROXY


async def test_the_sign_in_stays_direct_without_a_proxy() -> None:
    """Without a proxy the sign-in requests are the ones they always were."""
    session = MagicMock()
    session.callback_url = "http://ma.lan:8095/setup/callback"
    session.external = AsyncMock(return_value={"code": "abc"})
    token = {"refresh_token": "refresh", "access_token": "access"}
    session.mass.http_session.post = MagicMock(return_value=_response(200, token))
    session.mass.http_session.get = MagicMock(return_value=_response(200, {"id": "u1"}))
    await setup_flow._pkce_authenticate(session, "client", "authenticate")
    await setup_flow._get_profile(session, "access")
    assert "proxy" not in session.mass.http_session.post.call_args.kwargs
    assert "proxy" not in session.mass.http_session.get.call_args.kwargs


async def _run_setup(
    monkeypatch: pytest.MonkeyPatch, stored: dict[str, Any], asked: str
) -> tuple[dict[str, Any], dict[str, AsyncMock]]:
    """Run the whole flow with every step but the ones passing the proxy on stubbed."""
    stubs = {
        "_ask_proxy": AsyncMock(return_value=asked),
        "_pkce_authenticate": AsyncMock(
            return_value={"refresh_token": "refresh", "access_token": "access"}
        ),
        "_verify_account": AsyncMock(return_value="u1"),
        "_setup_playback": AsyncMock(),
        "_discard_pairing_dir": AsyncMock(),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(setup_flow, name, stub)
    session = MagicMock()
    session.context.setup_data = stored
    session.form = AsyncMock(return_value={})
    finished: dict[str, Any] = {}
    session.finish = AsyncMock(side_effect=finished.update)
    await setup_flow.run_setup(session)
    return finished, stubs


async def test_the_flow_asks_for_the_proxy_first_and_uses_it_throughout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proxy answered at the start reaches the sign-in, the lookup and the playback step."""
    finished, stubs = await _run_setup(monkeypatch, {CONF_PROXY_URL: "old:1"}, PROXY)
    # the stored proxy is what the question starts from
    assert stubs["_ask_proxy"].await_args_list[-1].args[1] == "old:1"
    assert stubs["_pkce_authenticate"].await_args_list[-1].kwargs["proxy_url"] == PROXY
    assert stubs["_verify_account"].await_args_list[-1].args[2] == PROXY
    assert stubs["_setup_playback"].await_args_list[-1].args[1][CONF_PROXY_URL] == PROXY
    assert finished[CONF_PROXY_URL] == PROXY
    assert finished[CONF_REFRESH_TOKEN_GLOBAL] == "refresh"


async def test_clearing_the_proxy_clears_the_stored_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty answer replaces the stored proxy instead of leaving it in force."""
    finished, stubs = await _run_setup(monkeypatch, {CONF_PROXY_URL: PROXY}, "")
    assert finished[CONF_PROXY_URL] is None
    assert stubs["_pkce_authenticate"].await_args_list[-1].kwargs["proxy_url"] == ""


async def test_the_playback_authorization_receives_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The librespot branch of the flow authorizes playback through the proxy."""
    authorize = AsyncMock(return_value="credentials")
    monkeypatch.setattr(setup_flow, "_authorize_playback", authorize)
    session = _flow_session({"playback_backend": "librespot"})
    setup_data: dict[str, Any] = {CONF_PROXY_URL: PROXY}
    await setup_flow._setup_playback(session, setup_data, "u1")
    authorize.assert_awaited_once_with(session, "u1", PROXY)


async def test_the_pairing_of_a_playback_attempt_uses_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairing through the Spotify app, and the browser variant, are handed the proxy."""
    monkeypatch.setattr(
        setup_flow, "get_librespot_binary", AsyncMock(return_value="/bin/librespot")
    )
    monkeypatch.setattr(setup_flow, "get_ip_addresses", AsyncMock(return_value=("10.0.0.2",)))
    pairing = MagicMock(return_value=object())
    monkeypatch.setattr(setup_flow, "librespot_credentials_via_pairing", pairing)
    browser = AsyncMock(return_value='{"username": "u1"}')
    monkeypatch.setattr(setup_flow, "_authorize_playback_via_browser", browser)
    session = MagicMock()
    session.mass.streams.publish_ip = "10.0.0.2"
    session.progress_until = AsyncMock(return_value='{"username": "u1"}')
    session.form = AsyncMock(
        side_effect=[
            {CONF_PLAYBACK_AUTH_METHOD: PLAYBACK_AUTH_APP},
            {CONF_PLAYBACK_AUTH_METHOD: PLAYBACK_AUTH_BROWSER},
        ]
    )
    await setup_flow._authorize_playback(session, None, PROXY)
    await setup_flow._authorize_playback(session, None, PROXY)
    pairing.assert_called_once_with("/bin/librespot", PAIRING_DEVICE_NAME, "10.0.0.2", PROXY)
    browser.assert_awaited_once_with(session, "/bin/librespot", PROXY)


async def test_a_playback_attempt_still_reports_its_failure_through_a_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proxy changes nothing about how a failed attempt is shown."""
    monkeypatch.setattr(
        setup_flow, "get_librespot_binary", AsyncMock(return_value="/bin/librespot")
    )
    monkeypatch.setattr(setup_flow, "get_ip_addresses", AsyncMock(return_value=("10.0.0.2",)))
    monkeypatch.setattr(
        setup_flow, "librespot_credentials_via_pairing", MagicMock(return_value=object())
    )
    session = MagicMock()
    session.mass.streams.publish_ip = "10.0.0.2"
    session.progress_until = AsyncMock(
        side_effect=[aiohttp.ClientConnectionError("proxy down"), '{"username": "u1"}']
    )
    session.form = AsyncMock(return_value={CONF_PLAYBACK_AUTH_METHOD: PLAYBACK_AUTH_APP})
    await setup_flow._authorize_playback(session, None, PROXY)
    assert session.form.await_args_list[1].kwargs["errors"] == {"base": "playback_auth_failed"}
    assert not isinstance(session.form.await_args_list[1].kwargs["errors"]["base"], SetupFlowError)


async def test_the_browser_authorization_of_playback_uses_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The code is exchanged, and the playback credential minted, through the proxy."""
    monkeypatch.setattr(setup_flow, "await_loopback_authorization", MagicMock())
    mint = AsyncMock(return_value='{"username": "u1"}')
    monkeypatch.setattr(setup_flow, "librespot_credentials_via_token", mint)
    session = MagicMock()
    session.external_until = AsyncMock(return_value={"code": "abc"})
    session.mass.http_session.post = MagicMock(
        return_value=_response(200, {"access_token": "playback-token"})
    )
    await setup_flow._authorize_playback_via_browser(session, "/bin/librespot", PROXY)
    assert session.mass.http_session.post.call_args.kwargs["proxy"] == PROXY
    mint.assert_awaited_once_with("/bin/librespot", "playback-token", PROXY)


async def test_the_developer_key_is_authorized_through_the_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signing in with the own developer key, and checking its account, use the proxy."""
    authenticate = AsyncMock(return_value={"refresh_token": "dev", "access_token": "dev-access"})
    verify = AsyncMock()
    monkeypatch.setattr(setup_flow, "_pkce_authenticate", authenticate)
    monkeypatch.setattr(setup_flow, "_verify_developer_account", verify)
    session = _flow_session({CONF_CLIENT_ID: "my-client"})
    session.finish = AsyncMock()
    setup_data: dict[str, Any] = {CONF_PROXY_URL: PROXY, "account_id": "u1"}
    await setup_flow._authorize_developer_key(session, setup_data, "")
    assert authenticate.await_args_list[-1].kwargs["proxy_url"] == PROXY
    verify.assert_awaited_once_with(session, "dev-access", "u1", PROXY)
