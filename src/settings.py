from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    OPENAI_API_KEY: str = ""
    LANGFUSE_BASE_URL: str = ""
    LANGFUSE_TRACING_ENVIRONMENT: str = "development"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""


settings = Settings()
