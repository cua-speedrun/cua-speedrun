"""Verify text keysyms on the actual RFB socket transport."""

import socket
import struct

from gym_anything.runtime.runners.vnc_utils import VNCConnection

from cua_speedrun.envs.keyboard import _type_vnc_text


def test_vnc_text_preserves_case_symbols_unicode_and_controls():
    text = "AaZz !@#$%()-_+/?: éÉ中文\t\n\r"
    controls = {"\t": 0xFF09, "\n": 0xFF0D, "\r": 0xFF0D}
    expected = []
    for char in text:
        code = controls.get(char, ord(char) if ord(char) <= 0xFF else 0x01000000 + ord(char))
        expected.extend([(4, 1, code), (4, 0, code)])

    sender, receiver = socket.socketpair()
    with sender, receiver:
        connection = VNCConnection("localhost", 0)
        connection._socket = sender
        _type_vnc_text(connection, text, delay=0)
        receiver.settimeout(1)
        received = bytearray()
        while len(received) < len(expected) * 8:
            received.extend(receiver.recv(len(expected) * 8 - len(received)))
        assert list(struct.iter_unpack("!BBxxI", received)) == expected
