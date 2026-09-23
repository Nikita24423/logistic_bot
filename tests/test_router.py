import os
import pytest
import httpx

from omnot.config import OmnotConfig
from omnot.router import OmnotRouter, AllProvidersExhausted


@pytest.fixture
def mock_config():
    os.environ["_OMNOT_MOCK_KEY"] = "test-key"
    config = OmnotConfig.from_dict({
        "providers": [
            {
                "name": "mock",
                "api_key_env": "_OMNOT_MOCK_KEY",
                "base_url": "https://mock.test",
                "models": ["test-model"],
                "rpm_limit": 100,
                "priority": 1,
            }
        ],
        "timeout": 5.0,
    })
    yield config
    del os.environ["_OMNOT_MOCK_KEY"]


def test_router_init(mock_config):
    router = OmnotRouter(mock_config)
    status = router.status()
    assert "mock" in status
    assert status["mock"]["models"] == ["test-model"]


@pytest.mark.asyncio
async def test_all_providers_exhausted_when_none_configured():
    config = OmnotConfig.from_dict({"providers": []})
    async with OmnotRouter(config) as router:
        with pytest.raises(AllProvidersExhausted):
            await router.chat([{"role": "user", "content": "test"}])
