import sqlite3
from pathlib import Path


def init_db(db_path: Path) -> None:
    """Inicializa la tabla de segmentos y el índice de búsqueda full-text FTS5."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_rel_path TEXT NOT NULL,
                subject TEXT NOT NULL,
                lesson_title TEXT NOT NULL,
                start_sec REAL NOT NULL,
                end_sec REAL NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                text,
                lesson_title,
                subject,
                content='segments',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
                INSERT INTO segments_fts(rowid, text, lesson_title, subject)
                VALUES (new.id, new.text, new.lesson_title, new.subject);
            END;
            """
        )
        conn.commit()


def replace_lesson_segments(
    db_path: Path,
    video_rel_path: str,
    subject: str,
    lesson_title: str,
    segments: list[dict],
) -> None:
    """Reemplaza los segmentos de una clase de forma idempotente."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM segments WHERE video_rel_path = ?",
            (video_rel_path,),
        ).fetchall()
        for (row_id,) in rows:
            conn.execute(
                "INSERT INTO segments_fts(segments_fts, rowid, text, lesson_title, subject) VALUES('delete', ?, '', '', '')",
                (row_id,),
            )
        conn.execute("DELETE FROM segments WHERE video_rel_path = ?", (video_rel_path,))

        for seg in segments:
            conn.execute(
                """
                INSERT INTO segments (video_rel_path, subject, lesson_title, start_sec, end_sec, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    video_rel_path,
                    subject,
                    lesson_title,
                    float(seg["start"]),
                    float(seg["end"]),
                    seg["text"].strip(),
                ),
            )
        conn.commit()


def search_segments(db_path: Path, query: str, limit: int = 50) -> list[dict]:
    """Busca coincidencias en el índice FTS5 ordenadas por relevancia."""
    clean_query = query.strip()
    if not clean_query or not db_path.exists():
        return []

    # Escapa comillas dobles para evitar errores de sintaxis FTS5
    safe_terms = " ".join(f'"{tok.replace(chr(34), "")}"' for tok in clean_query.split() if tok.strip())
    if not safe_terms:
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT
                s.id,
                s.video_rel_path,
                s.subject,
                s.lesson_title,
                s.start_sec,
                s.end_sec,
                s.text
            FROM segments_fts f
            JOIN segments s ON s.id = f.rowid
            WHERE segments_fts MATCH ?
            ORDER BY bm25(segments_fts)
            LIMIT ?
            """,
            (safe_terms, limit),
        )
        return [dict(r) for r in cursor.fetchall()]