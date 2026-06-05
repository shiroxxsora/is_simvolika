ALTER TABLE tg_user_profiles
    ADD COLUMN IF NOT EXISTS specialist_education TEXT;

ALTER TABLE tg_user_profiles
    ADD COLUMN IF NOT EXISTS specialist_qualification TEXT;

ALTER TABLE tg_user_profiles
    ADD COLUMN IF NOT EXISTS specialist_additional_training TEXT;

ALTER TABLE tg_user_profiles
    ADD COLUMN IF NOT EXISTS specialist_position TEXT;

ALTER TABLE tg_user_profiles
    ADD COLUMN IF NOT EXISTS specialist_research_interests TEXT;

ALTER TABLE tg_user_profiles
    ADD COLUMN IF NOT EXISTS specialist_experience_years TEXT;

ALTER TABLE tg_user_profiles
    ADD COLUMN IF NOT EXISTS report_basis TEXT;

