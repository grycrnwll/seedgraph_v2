import sqlite3

from sqlalchemy import text
from sqlmodel import Session, create_engine

from seedgraph.db.adapter import raw_conn


def test_raw_conn_shares_session_transaction(tmp_path):
    db = (tmp_path / "adapter.db").as_posix()
    engine = create_engine(f"sqlite:///{db}")
    # create the table in its own committed transaction (no DDL-rollback ambiguity)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")

    with Session(engine) as session:
        raw = raw_conn(session)
        assert isinstance(raw, sqlite3.Connection)
        # a raw INSERT through the bound connection...
        raw.execute("INSERT INTO t (id, v) VALUES (1, 'x')")
        # ...is visible to the same session before commit (same transaction)
        count = session.execute(text("SELECT COUNT(*) FROM t")).scalar_one()
        assert count == 1
        session.rollback()

    # ...and is rolled back with the transaction
    with Session(engine) as session2:
        count2 = session2.execute(text("SELECT COUNT(*) FROM t")).scalar_one()
        assert count2 == 0
