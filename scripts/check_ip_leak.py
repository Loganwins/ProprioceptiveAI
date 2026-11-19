#!/usr/bin/env python3
"""Pre-publish IP-leak gate for the proprioceptive-lens client wheel.

The client must stay IP-free: the proprietary steering vectors / dark-head theta
live server-side and only their OUTPUTS cross the wire. A 9+ float array in the
client is the smoking gun for inlined IP. Run this on dist/*.whl before publish;
it exits 1 (fails the publish) if any such array is found.

Usage:  python scripts/check_ip_leak.py [dist-dir]
"""
import re
import sys
import zipfile
import glob
import os

# 9+ comma-separated floats = a steering vector / theta (the IP).
RX = re.compile(r"-?[0-9]+(?:\.[0-9]+)?\s*(?:,\s*-?[0-9]+(?:\.[0-9]+)?){8,}")


def main():
    dist_dir = sys.argv[1] if len(sys.argv) > 1 else "dist"
    wheels = glob.glob(os.path.join(dist_dir, "*.whl"))
    if not wheels:
        print("  no wheels found in", dist_dir)
        return 0
    bad = False
    for whl in wheels:
        with zipfile.ZipFile(whl) as z:
            for n in z.namelist():
                if not n.endswith(".py"):
                    continue
                txt = z.read(n).decode("utf-8", "replace")
                m = RX.search(txt)
                if m:
                    line = txt[: m.start()].count("\n") + 1
                    print("  IP-LEAK SUSPECT  %s  (line %d)" % (n, line))
                    bad = True
    if bad:
        print("  FAIL: the client wheel contains a 9+ float array (steering vector / theta).")
        print("  The IP must stay server-side. Fix the source, do NOT publish.")
        return 1
    print("  PASS: no IP (9+ float array) found in the client wheel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
