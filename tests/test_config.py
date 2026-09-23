import os
from omnot.config import OmnotConfig, ProviderConfig


def test_provider_config_defaults():
    p = ProviderConfig(
        name="test",
        api_key_env="TEST_KEY",
        base_url="https://example.com",
        models=["model-a"],
    )
    assert p.rpm_limit == 0
    assert p.priority == 0
    assert p.enabled is True


def test_provider_not_configured_without_env():
    p = ProviderConfig(
        name="test",
        api_key_env="NONEXISTENT_KEY_12345",
        base_url="https://example.com",
        models=["model-a"],
    )
    assert p.is_configured is False


def test_config_from_dict():
    data = {
        "providers": [
            {
                "name": "test",
                "api_key_env": "TEST_KEY",
                "base_url": "https://example.com",
                "models": ["model-a"],
                "priority": 1,
            }
        ],
        "fallback_strategy": "priority",
        "max_retries": 5,
    }
    config = OmnotConfig.from_dict(data)
    assert len(config.providers) == 1
    assert config.providers[0].name == "test"
    assert config.fallback_strategy == "priority"
    assert config.max_retries == 5


def test_active_providers_sorted_by_priority():
    os.environ["_OMNOT_TEST_A"] = "key-a"
    os.environ["_OMNOT_TEST_B"] = "key-b"
    try:
        config = OmnotConfig.from_dict({
            "providers": [
                {
                    "name": "low-prio",
                    "api_key_env": "_OMNOT_TEST_A",
                    "base_url": "https://a.com",
                    "models": ["m"],
                    "priority": 10,
                },
                {
                    "name": "high-prio",
                    "api_key_env": "_OMNOT_TEST_B",
                    "base_url": "https://b.com",
                    "models": ["m"],
                    "priority": 1,
                },
            ]
        })
        active = config.get_active_providers()
        assert active[0].name == "high-prio"
        assert active[1].name == "low-prio"
    finally:
        del os.environ["_OMNOT_TEST_A"]
        del os.environ["_OMNOT_TEST_B"]
