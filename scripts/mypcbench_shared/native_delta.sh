#!/usr/bin/env bash
# Port the pinned persona filesystem to a native desktop sandbox.
set -euo pipefail
R="${R:?rootfs required}"
grep -q 'VERSION_CODENAME=noble' "$R/etc/os-release"
test -d "$R/home/user"
rm -f "$R/etc/resolv.conf"
cp /etc/resolv.conf "$R/etc/resolv.conf"
printf '# Native sandbox root; no guest disk mounts.\n' > "$R/etc/fstab"
mount --rbind /dev "$R/dev"
mount -t proc proc "$R/proc"
mount --rbind /sys "$R/sys"
mount -t tmpfs tmpfs "$R/run"
cleanup() {
    umount -R "$R/run" "$R/proc" "$R/sys" "$R/dev" 2>/dev/null || true
}
trap cleanup EXIT
# Avoid starting daemons while preparing an unbooted filesystem.
test ! -e "$R/usr/sbin/policy-rc.d"
printf '#!/bin/sh\nexit 101\n' > "$R/usr/sbin/policy-rc.d"
chmod 755 "$R/usr/sbin/policy-rc.d"
cat > "$R/etc/apt/sources.list.d/cs-native.list" <<'EOF'
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20260716T000000Z noble main universe
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20260716T000000Z noble-updates main universe
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20260716T000000Z noble-security main universe
EOF
opts=(-o Dir::Etc::sourcelist=sources.list.d/cs-native.list -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0)
chroot "$R" apt-get "${opts[@]}" update -q
DEBIAN_FRONTEND=noninteractive chroot "$R" apt-get "${opts[@]}" install -y --no-install-recommends xserver-xorg-video-dummy xdotool scrot python3-xlib python3-tk python3-pip
rm "$R/usr/sbin/policy-rc.d" "$R/etc/apt/sources.list.d/cs-native.list"
chroot "$R" dpkg-query -W > "$R/var/lib/cua-native-packages.txt"
# Fold the image's SQLite journals into the browser databases before Firefox
# opens concurrent connections. Preserve all seeded rows and schema.
chroot "$R" runuser -u user -- python3 - <<'PYSQLITE'
from pathlib import Path
import hashlib
import sqlite3

profile = Path('/home/user/.mozilla/firefox/benchmark.default-release')
for name in ('places.sqlite', 'favicons.sqlite'):
    connection = sqlite3.connect(profile / name)
    before = hashlib.sha256('\n'.join(connection.iterdump()).encode()).hexdigest()
    result = connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
    if result[0] != 0:
        raise RuntimeError(f'{name}: could not checkpoint the seeded database: {result}')
    after = hashlib.sha256('\n'.join(connection.iterdump()).encode()).hexdigest()
    if before != after:
        raise RuntimeError(f'{name}: seeded browser data changed')
    connection.close()
    print('Browser database checkpointed:', name, before, flush=True)
PYSQLITE
# Preserve the image's own GDM session and seeded browser profile.
grep -q 'AutomaticLoginEnable.*=.*True' "$R/etc/gdm3/custom.conf"
grep -q 'AutomaticLogin.*=.*user' "$R/etc/gdm3/custom.conf"
grep -q 'WaylandEnable.*=.*false' "$R/etc/gdm3/custom.conf"
echo 'user ALL=(ALL) NOPASSWD:ALL' > "$R/etc/sudoers.d/99-cua-native"
chmod 440 "$R/etc/sudoers.d/99-cua-native"
for unit in NetworkManager.service NetworkManager-wait-online.service systemd-resolved.service systemd-timesyncd.service systemd-udevd.service systemd-udevd-control.socket systemd-udevd-kernel.socket systemd-oomd.service unattended-upgrades.service; do
    ln -sfn /dev/null "$R/etc/systemd/system/$unit"
done
cat > "$R/etc/X11/xorg.conf" <<'EOF'
Section "Device"
    Identifier "NativeDummy"
    Driver "dummy"
    VideoRam 256000
EndSection
Section "Monitor"
    Identifier "NativeMonitor"
    HorizSync 30.0-80.0
    VertRefresh 50.0-75.0
    Modeline "1280x800" 83.50 1280 1352 1480 1680 800 803 809 831 -hsync +vsync
EndSection
Section "Screen"
    Identifier "NativeScreen"
    Device "NativeDummy"
    Monitor "NativeMonitor"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1280x800"
        Virtual 1280 800
    EndSubSection
EndSection
EOF
# The upstream image autostarts Firefox independently of the seeded app units.
# A native boot can reach GNOME before those HTTP listeners are ready, leaving
# restored tabs on connection errors. Preserve the original profile/arguments,
# but delay that original launch until every image-owned app answers.
cat > "$R/usr/local/bin/cua-wait-mypcbench-apps" <<'PYWAIT'
#!/usr/bin/python3
import http.cookiejar
import os
import sys
import time
import urllib.request

deadline = time.monotonic() + 600
pending = set(range(3001, 3018))
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
while pending and time.monotonic() < deadline:
    for port in sorted(pending):
        try:
            with opener.open(f"http://localhost:{port}/", timeout=2) as response:
                if response.status == 200:
                    pending.remove(port)
        except Exception:
            pass
    if pending:
        time.sleep(1)
if pending:
    raise SystemExit("MyPCBench apps unavailable before browser startup: " + str(sorted(pending)))
os.execvp(sys.argv[1], sys.argv[1:])
PYWAIT
chmod 755 "$R/usr/local/bin/cua-wait-mypcbench-apps"
python3 - "$R/home/user/.config/autostart/firefox.desktop" <<'PYAUTOSTART'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
assert sum(line.startswith("Exec=") for line in text.splitlines()) == 1
path.write_text(text.replace("Exec=", "Exec=/usr/local/bin/cua-wait-mypcbench-apps ", 1))
PYAUTOSTART
echo MYPCBENCH_NATIVE_DELTA_OK
