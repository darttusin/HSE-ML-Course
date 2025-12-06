from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter=".",
        case_sensitive=False,
        extra="ignore",
    )
    scaler_model_path: str = "models/hw1_scaler.pkl"
    ridge_model_path: str = "models/hw1_ridge_model.pkl"
    onehot_encoder_path: str = "models/hw1_one_hot_encoder.pkl"

    numeric_features: list[str] = [
        "year",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
    ]
    categorical_features: list[str] = [
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "seats",
    ]


settings = Settings()  # type: ignore
