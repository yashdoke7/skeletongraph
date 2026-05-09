from skeletongraph.config import SGConfig


def test_cli_provider_preset_sets_provider_models():
    config = SGConfig()

    config.apply_cli_provider_preset("openai")

    assert config.cli_provider == "openai"
    assert config.get_cli_model_for_tier("slm") == "gpt-5.4-mini"
    assert config.get_cli_model_for_tier("mlm") == "gpt-5.5"
    assert config.get_cli_model_for_tier("llm") == "gpt-5.5"
    assert config.get_cli_key_envs() == ["OPENAI_API_KEY"]


def test_cli_provider_key_status_uses_env(monkeypatch):
    config = SGConfig()
    config.apply_cli_provider_preset("anthropic")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not config.cli_api_key_configured()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert config.cli_api_key_configured()


def test_ide_and_cli_model_names_are_separate():
    config = SGConfig(
        slm_model="IDE small",
        mlm_model="IDE medium",
        llm_model="IDE large",
        cli_slm_model="api-small",
        cli_mlm_model="api-medium",
        cli_llm_model="api-large",
    )

    assert config.get_model_for_tier("mlm") == "IDE medium"
    assert config.get_cli_model_for_tier("mlm") == "api-medium"


def test_local_cli_provider_does_not_require_api_key():
    config = SGConfig()

    config.apply_cli_provider_preset("local")

    assert config.cli_provider == "local"
    assert config.get_cli_key_envs() == []
    assert config.cli_api_key_configured()
    assert config.get_cli_api_base() == "http://localhost:11434"
    assert config.get_cli_model_for_tier("mlm").startswith("ollama/")
