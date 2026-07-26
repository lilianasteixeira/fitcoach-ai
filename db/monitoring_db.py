"""
Helper functions to save conversations and feedback to the monitoring
database (Postgres), used by the Streamlit app.
"""

import os

import psycopg

DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://fitcoach:fitcoach@postgres:5432/fitcoach"
)


def get_connection():
    return psycopg.connect(DB_URL)


def log_conversation(question, answer, model, search_strategy, prompt_template, latency_seconds, topic=None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations
                    (question, answer, model, search_strategy, prompt_template, latency_seconds, topic)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (question, answer, model, search_strategy, prompt_template, latency_seconds, topic),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
    return conversation_id


def save_feedback(conversation_id: int, feedback: int):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET feedback = %s WHERE id = %s;",
                (feedback, conversation_id),
            )
        conn.commit()
