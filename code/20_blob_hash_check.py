"""Step 20 - Check that GitSkills identifies a file by its git blob hash.

    python 20_blob_hash_check.py --db path/to/agent_skills_sample.db [--n 200]

Step 17 joins our pairs with the copy network of Seddik through git blob hashes. This works only if the
file_sha of GitSkills is the SHA-1 of the git blob of the file content, i.e. sha1("blob <size>\\0" + bytes).
The script draws --n files at random (fixed seed) from the artifacts table, recomputes the git blob hash of
their content and counts how many equal file_sha. Works on the official sample and on the full release.
"""
import argparse
import hashlib
import sqlite3

from common import SEED

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True, help="GitSkills SQLite database (sample or full release)")
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()

con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
rows = con.execute(f"SELECT file_sha, content FROM artifacts WHERE content IS NOT NULL AND content != '' "
                   f"ORDER BY substr(file_sha, 1 + {SEED % 30}, 10) LIMIT ?", (args.n,)).fetchall()
match = 0
for sha, content in rows:
    data = content.encode("utf-8")
    match += hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == sha
print(f"git blob SHA-1 of the content equals file_sha for {match} of {len(rows)} files")
