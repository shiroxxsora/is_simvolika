CREATE TABLE IF NOT EXISTS tg_user_profiles (
    telegram_user_id BIGINT PRIMARY KEY,
    full_name TEXT,
    specialist_education TEXT,
    specialist_qualification TEXT,
    specialist_additional_training TEXT,
    specialist_position TEXT,
    specialist_research_interests TEXT,
    specialist_experience_years TEXT,
    report_basis TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS tg_user_profiles_full_name_idx
    ON tg_user_profiles (full_name);

