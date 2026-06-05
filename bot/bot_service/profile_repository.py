from __future__ import annotations

from dataclasses import dataclass

try:
    from psycopg import connect
    from psycopg import errors as pg_errors
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'psycopg'. Install bot requirements: pip install -r bot/requirements.txt"
    ) from e


DEFAULT_DISPLAY_NAME = "[Не заданное Имя]"


@dataclass(frozen=True)
class Profile:
    telegram_user_id: int
    full_name: str | None
    specialist_education: str | None = None
    specialist_qualification: str | None = None
    specialist_additional_training: str | None = None
    specialist_position: str | None = None
    specialist_research_interests: str | None = None
    specialist_experience_years: str | None = None
    report_basis: str | None = None

    @property
    def display_name(self) -> str:
        name = (self.full_name or "").strip()
        return name if name else DEFAULT_DISPLAY_NAME


class ProfileRepository:
    def __init__(self, postgres_dsn: str) -> None:
        self.postgres_dsn = postgres_dsn

    def get_profile(self, telegram_user_id: int) -> Profile:
        try:
            with connect(self.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT full_name, specialist_education, specialist_qualification, "
                        "specialist_additional_training, specialist_position, specialist_research_interests, "
                        "specialist_experience_years, report_basis "
                        "FROM tg_user_profiles WHERE telegram_user_id = %s",
                        (telegram_user_id,),
                    )
                    row = cur.fetchone()
                    return Profile(
                        telegram_user_id=telegram_user_id,
                        full_name=(row[0] if row else None),
                        specialist_education=(row[1] if row else None),
                        specialist_qualification=(row[2] if row else None),
                        specialist_additional_training=(row[3] if row else None),
                        specialist_position=(row[4] if row else None),
                        specialist_research_interests=(row[5] if row else None),
                        specialist_experience_years=(row[6] if row else None),
                        report_basis=(row[7] if row else None),
                    )
        except pg_errors.UndefinedTable:
            return Profile(telegram_user_id=telegram_user_id, full_name=None)

    def upsert_full_name(self, telegram_user_id: int, full_name: str) -> tuple[Profile, bool]:
        """Returns (profile, changed)."""
        cleaned = " ".join((full_name or "").split()).strip()
        if not cleaned:
            return self.get_profile(telegram_user_id), False

        try:
            with connect(self.postgres_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT full_name FROM tg_user_profiles WHERE telegram_user_id = %s",
                        (telegram_user_id,),
                    )
                    existing_row = cur.fetchone()
                    existing = (existing_row[0] if existing_row else None)

                    cur.execute(
                        """
                        INSERT INTO tg_user_profiles (telegram_user_id, full_name)
                        VALUES (%s, %s)
                        ON CONFLICT (telegram_user_id) DO UPDATE
                        SET full_name = EXCLUDED.full_name,
                            updated_at = NOW()
                        """,
                        (telegram_user_id, cleaned),
                    )

                    changed = (existing or "").strip() != cleaned
                    return Profile(telegram_user_id=telegram_user_id, full_name=cleaned), changed
        except pg_errors.UndefinedTable:
            # Migrations may not be applied yet; behave as "not set" without crashing.
            return Profile(telegram_user_id=telegram_user_id, full_name=None), False

    def upsert_specialist_fields(
        self,
        telegram_user_id: int,
        *,
        education: str | None = None,
        qualification: str | None = None,
        additional_training: str | None = None,
        position: str | None = None,
        research_interests: str | None = None,
        experience_years: str | None = None,
        report_basis: str | None = None,
    ) -> tuple[Profile, bool]:
        """Upsert specialist/report fields. Returns (profile, changed)."""
        def clean(v: str | None) -> str | None:
            if v is None:
                return None
            s = " ".join(v.split()).strip()
            return s or None

        vals = {
            "education": clean(education),
            "qualification": clean(qualification),
            "additional_training": clean(additional_training),
            "position": clean(position),
            "research_interests": clean(research_interests),
            "experience_years": clean(experience_years),
            "report_basis": clean(report_basis),
        }
        if not any(v is not None for v in vals.values()):
            return self.get_profile(telegram_user_id), False

        try:
            before = self.get_profile(telegram_user_id)
            with connect(self.postgres_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO tg_user_profiles (
                            telegram_user_id,
                            full_name,
                            specialist_education,
                            specialist_qualification,
                            specialist_additional_training,
                            specialist_position,
                            specialist_research_interests,
                            specialist_experience_years,
                            report_basis
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (telegram_user_id) DO UPDATE
                        SET specialist_education = COALESCE(EXCLUDED.specialist_education, tg_user_profiles.specialist_education),
                            specialist_qualification = COALESCE(EXCLUDED.specialist_qualification, tg_user_profiles.specialist_qualification),
                            specialist_additional_training = COALESCE(EXCLUDED.specialist_additional_training, tg_user_profiles.specialist_additional_training),
                            specialist_position = COALESCE(EXCLUDED.specialist_position, tg_user_profiles.specialist_position),
                            specialist_research_interests = COALESCE(EXCLUDED.specialist_research_interests, tg_user_profiles.specialist_research_interests),
                            specialist_experience_years = COALESCE(EXCLUDED.specialist_experience_years, tg_user_profiles.specialist_experience_years),
                            report_basis = COALESCE(EXCLUDED.report_basis, tg_user_profiles.report_basis),
                            updated_at = NOW()
                        """,
                        (
                            telegram_user_id,
                            (before.full_name or None),
                            vals["education"],
                            vals["qualification"],
                            vals["additional_training"],
                            vals["position"],
                            vals["research_interests"],
                            vals["experience_years"],
                            vals["report_basis"],
                        ),
                    )
            after = self.get_profile(telegram_user_id)
            changed = (
                (before.specialist_education or "") != (after.specialist_education or "")
                or (before.specialist_qualification or "") != (after.specialist_qualification or "")
                or (before.specialist_additional_training or "") != (after.specialist_additional_training or "")
                or (before.specialist_position or "") != (after.specialist_position or "")
                or (before.specialist_research_interests or "") != (after.specialist_research_interests or "")
                or (before.specialist_experience_years or "") != (after.specialist_experience_years or "")
                or (before.report_basis or "") != (after.report_basis or "")
            )
            return after, changed
        except pg_errors.UndefinedTable:
            return self.get_profile(telegram_user_id), False

