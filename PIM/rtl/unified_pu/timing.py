#!/usr/bin/env python3
"""Critical path of every mapped module of a run (ABC stime after mapping; NanGate45 typical, no wire load)."""
import re, sys
for f, what in zip(sys.argv[1::2], sys.argv[2::2]):
    mods, rows, final = [], [], False
    for l in open(f):
        if re.match(r"\d+\. Executing ABC pass", l): final, mods = True, []
        m = re.search(r"Extracting gate netlist of module `\\?(.+?)'", l)
        if m and final: mods.append(m.group(1))
        m = re.search(r"ABC: WireLoad.*Gates =\s*(\d+).*Delay =\s*([\d.]+) ps", l)
        if m: g, d = m.group(1), float(m.group(2))
        m = re.search(r"ABC: Start-point = \S+ \((.+?)\)\.\s+End-point = \S+ \((.+?)\)", l)
        if m: rows.append((g, d, m.group(1), m.group(2)))
    print(what)
    for mod, (g, d, a, b) in zip(mods, rows):
        if d > 500: print(f"   {mod:10} {int(g):6d} gates  {d:7.0f} ps   {a[:40]} -> {b[:40]}")
