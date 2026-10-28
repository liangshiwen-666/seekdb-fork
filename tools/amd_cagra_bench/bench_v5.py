#!/usr/bin/env python3
"""V5 benchmark: build time (blocking CREATE VECTOR INDEX + first query),
Recall@10/@100 and QPS over --nq single-row queries, like the official flow.

Usage: bench_v5.py --data DIR --binary PATH --bridge DIR --port P --tag NAME [--nq 4096]
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
import numpy as np
import pymysql

def log(*a):
    print("[bench]", *a, flush=True)

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def mysql_exec(port, sql, many=None):
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root", autocommit=True)
    try:
        cur = conn.cursor()
        if many is not None:
            cur.executemany(sql, many)
        else:
            cur.execute(sql)
            rows = cur.fetchall() if cur.description else []
            return rows
    finally:
        conn.close()

def vec_lit(v):
    return "[" + ",".join("%.6f" % x for x in v) + "]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--port", type=int, default=2881)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--nq", type=int, default=0, help="queries for recall/QPS (0=all)")
    ap.add_argument("--skip-qps", action="store_true")
    ap.add_argument("--out", default="/root/work/bench/results")
    ap.add_argument("--basedir", default="")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    n, dim, nq_total = meta["n"], meta["dim"], meta["nq"]
    base = np.fromfile(os.path.join(args.data, "base.f32"), dtype=np.float32).reshape(n, dim)
    queries = np.fromfile(os.path.join(args.data, "query.f32"), dtype=np.float32).reshape(nq_total, dim)
    gtp = sorted(glob.glob(os.path.join(args.data, "gt_*x100.i32")))[0]
    gt100 = np.fromfile(gtp, dtype=np.int32).reshape(nq_total, 100)

    nq = args.nq if args.nq > 0 else nq_total
    nq = min(nq, nq_total)

    sh(["pkill", "-f", "seekdb --base-dir=/root/work/bench/work"])
    for _ in range(30):
        if not sh(["pgrep", "-f", "seekdb --base-dir=/root/work/bench/work"]).stdout.strip():
            break
        time.sleep(0.5)
    sh(["pkill", "-9", "-f", "seekdb --base-dir=/root/work/bench/work"])
    time.sleep(1)
    base_dir = args.basedir or ("/root/work/bench/work/%s-%s" % (args.tag, time.strftime("%H%M%S")))
    if os.path.isdir(base_dir):
        sh(["pkill", "-f", "base-dir=" + base_dir])
        shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)
    out_f = open(base_dir + ".out", "w")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = args.bridge + ":/opt/hipvs/lib:/opt/rocm/lib"
    proc = subprocess.Popen([args.binary, "--base-dir=" + base_dir, "--port=%d" % args.port,
                             "--nodaemon", "--parameter", "_memory_budget=32G",
                             "--parameter", "datafile_size=20G", "--parameter", "datafile_maxsize=80G",
                             "--parameter", "log_disk_size=10G", "--log-level", "INFO"],
                            stdout=out_f, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < 180:
        try:
            mysql_exec(args.port, "select 1")
            break
        except Exception:
            time.sleep(1)
    log("observer ready in %.1fs" % (time.time() - t0))

    conn = pymysql.connect(host="127.0.0.1", port=args.port, user="root", autocommit=True)
    cur = conn.cursor()
    for stmt in ("set session ob_query_timeout = 600000000",
                 "set session ob_trx_timeout = 600000000"):
        try:
            cur.execute(stmt)
        except Exception as e:
            log("warn:", stmt, e)
    cur.execute("drop database if exists vec")
    cur.execute("create database vec")
    cur.execute("use vec")
    cur.execute("create table items(id bigint primary key, v vector(%d))" % dim)
    t_imp = time.time()
    batch = 500
    for s in range(0, n, batch):
        rows = [(i, vec_lit(base[i])) for i in range(s, min(s + batch, n))]
        cur.executemany("insert into items values (%s, %s)", rows)
    t_imp = time.time() - t_imp
    cur.execute("select count(*) from items")
    cnt = cur.fetchone()[0]
    log("imported %d rows in %.1fs" % (cnt, t_imp))

    # timed build: blocking DDL + first query
    T0 = time.time()
    cur.execute("create vector index idx_items on items(v) with "
                "(distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64)")
    T1 = time.time()
    q0 = vec_lit(queries[0])
    cur.execute("select id from items order by l2_distance(v, '%s') approximate limit 10" % q0)
    rows0 = cur.fetchall()
    T2 = time.time()
    log("ddl=%.2fs  first_query=%.3fs  build_total=%.2fs  rows=%d" % (T1 - T0, T2 - T1, T2 - T0, len(rows0)))

    res = dict(tag=args.tag, data=os.path.basename(args.data), n=n, dim=dim, nq=nq,
               import_s=round(t_imp, 2), ddl_s=round(T1 - T0, 2), first_query_s=round(T2 - T1, 3),
               build_s=round(T2 - T0, 2), first_rows=len(rows0),
               binary=args.binary, bridge=args.bridge)

    # recall + QPS for top10 / top100
    if not args.skip_qps:
        for k, gt_k in ((10, gt100[:nq, :10]), (100, gt100[:nq])):
            got = []
            t = time.time()
            for i in range(nq):
                cur.execute("select id from items order by l2_distance(v, '%s') approximate limit %d" % (vec_lit(queries[i]), k))
                got.append([r[0] for r in cur.fetchall()])
            dt = time.time() - t
            hit = 0
            for i in range(nq):
                hit += len(set(got[i]) & set(gt_k[i].tolist()))
            rec = hit / float(nq * k)
            qps = nq / dt
            res["recall@%d" % k] = round(rec, 4)
            res["qps_top%d" % k] = round(qps, 1)
            log("top%d: recall=%.4f qps=%.1f over %d queries" % (k, rec, qps, nq))
    res["gpu_log"] = subprocess.run(["bash", "-c",
        "grep -o 'devbridge\\][^\\\"]*\\|OBCUVS02[^\\\"]*' %s/log/seekdb.log 2>/dev/null | head -20" % base_dir],
        capture_output=True, text=True).stdout.strip().splitlines()
    print(json.dumps(res, indent=2), flush=True)

    with open(os.path.join(args.out, "%s-%s.json" % (args.tag, time.strftime("%H%M%S"))), "w") as f:
        json.dump(res, f, indent=2)
    conn.close()
    proc.terminate()
    try:
        proc.wait(timeout=90)
    except Exception:
        proc.kill()

if __name__ == "__main__":
    main()
