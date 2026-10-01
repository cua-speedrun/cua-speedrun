#!/usr/bin/env bash
set -euo pipefail
R="${R:?rootfs required}"
codename=$(sed -n 's/^VERSION_CODENAME=//p' "$R/etc/os-release")
case "$codename" in jammy|noble) ;; *) echo "Unsupported desktop: $codename" >&2; exit 1 ;; esac
test -d "$R/home/user"
rm -f "$R/etc/resolv.conf"
cp /etc/resolv.conf "$R/etc/resolv.conf"
printf '# Native sandbox root; no guest disk mounts.\n' > "$R/etc/fstab"
mount --rbind /dev "$R/dev"
mount -t proc proc "$R/proc"
mount --rbind /sys "$R/sys"
mount -t tmpfs tmpfs "$R/run"
cleanup() { umount -R "$R/run" "$R/proc" "$R/sys" "$R/dev" 2>/dev/null || true; }
trap cleanup EXIT
test ! -e "$R/usr/sbin/policy-rc.d"
printf '#!/bin/sh\nexit 101\n' > "$R/usr/sbin/policy-rc.d"
chmod 755 "$R/usr/sbin/policy-rc.d"
cat > "$R/etc/apt/sources.list.d/cs-native.list" <<EOF
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20260716T000000Z $codename main universe
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20260716T000000Z $codename-updates main universe
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/20260716T000000Z $codename-security main universe
EOF
opts=(-o Dir::Etc::sourcelist=sources.list.d/cs-native.list -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0)
chroot "$R" apt-get "${opts[@]}" update -q
DEBIAN_FRONTEND=noninteractive chroot "$R" apt-get "${opts[@]}" install -y --no-install-recommends xserver-xorg-video-dummy xdotool scrot python3-xlib python3-tk
rm "$R/usr/sbin/policy-rc.d" "$R/etc/apt/sources.list.d/cs-native.list"
chroot "$R" dpkg-query -W > "$R/var/lib/cua-native-packages.txt"
grep -q 'AutomaticLoginEnable.*=.*True' "$R/etc/gdm3/custom.conf"
grep -q 'AutomaticLogin.*=.*user' "$R/etc/gdm3/custom.conf"
grep -q 'WaylandEnable.*=.*false' "$R/etc/gdm3/custom.conf"
echo 'user ALL=(ALL) NOPASSWD:ALL' > "$R/etc/sudoers.d/99-cua-native"
chmod 440 "$R/etc/sudoers.d/99-cua-native"
for unit in acpid.path acpid.service acpid.socket NetworkManager.service NetworkManager-wait-online.service systemd-resolved.service systemd-timesyncd.service systemd-udevd.service systemd-udevd-control.socket systemd-udevd-kernel.socket systemd-oomd.service unattended-upgrades.service; do
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
    Modeline "1920x1080" 148.50 1920 2008 2052 2200 1080 1084 1089 1125 +hsync +vsync
EndSection
Section "Screen"
    Identifier "NativeScreen"
    Device "NativeDummy"
    Monitor "NativeMonitor"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1920x1080"
        Virtual 1920 1080
    EndSubSection
EndSection
EOF
echo OSWORLD2_NATIVE_DELTA_OK
