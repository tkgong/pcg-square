#!/usr/bin/env python3
"""TCP small-message round-trip latency (the alpha of the loopback / DC tiers).
Mirrors the protocol's party-serialized exchange: A sends MSG bytes, B echoes
MSG bytes back; one iteration = one full RTT. TCP_NODELAY on, blocking sockets
-- same discipline as emp NetIO.

server:  python3 alpha_tcp.py server <port>
client:  python3 alpha_tcp.py client <host> <port> [msg=4096] [iters=2000]
Output (client): alpha_tcp,<host>,<msg>,<p50_us>,<p10_us>,<p90_us>
"""
import socket, sys, time, statistics

def recv_all(s, n):
    buf = b""
    while len(buf) < n:
        d = s.recv(n - len(buf))
        if not d:
            raise ConnectionError("peer closed")
        buf += d
    return buf

def server(port):
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", port)); ls.listen(1)
    c, _ = ls.accept()
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    hdr = recv_all(c, 16)
    msg, iters = int(hdr[:8]), int(hdr[8:])
    for _ in range(iters + 50):
        c.sendall(recv_all(c, msg))
    c.close()

def client(host, port, msg, iters):
    s = socket.socket(); s.connect((host, port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.sendall(b"%08d%08d" % (msg, iters))
    payload = b"x" * msg
    for _ in range(50):                              # warmup
        s.sendall(payload); recv_all(s, msg)
    rtt = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        s.sendall(payload)
        recv_all(s, msg)
        rtt.append((time.perf_counter_ns() - t0) / 1e3)
    rtt.sort()
    print("alpha_tcp,%s,%d,%.2f,%.2f,%.2f" %
          (host, msg, rtt[len(rtt)//2], rtt[len(rtt)//10], rtt[len(rtt)*9//10]))

if __name__ == "__main__":
    if sys.argv[1] == "server":
        server(int(sys.argv[2]))
    else:
        client(sys.argv[2], int(sys.argv[3]),
               int(sys.argv[4]) if len(sys.argv) > 4 else 4096,
               int(sys.argv[5]) if len(sys.argv) > 5 else 2000)
