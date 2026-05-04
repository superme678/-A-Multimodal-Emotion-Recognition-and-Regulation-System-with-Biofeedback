"""构造符合协议的 19 字节生理手环帧（与 sscom/sscom.py、webtest 解析一致）。"""


def _u16_be(value: int) -> tuple[int, int]:
    v = value & 0xFFFF
    return (v >> 8) & 0xFF, v & 0xFF


def build_physio_frame(
    gsr_raw: int,
    heart_rate: int,
    spo2: int,
    acc_raw_xyz: tuple[int, int, int] | None = None,
    gyro_raw_xyz: tuple[int, int, int] | None = None,
) -> bytes:
    """
    gsr_raw: 0..65535，对应解析式 gsr_raw = b1*256+b2
    acc/gyro_raw: 有符号 16 位原始值，与解析端 high*256+low 一致
    """
    if acc_raw_xyz is None:
        acc_raw_xyz = (0, 0, 0)
    if gyro_raw_xyz is None:
        gyro_raw_xyz = (0, 0, 0)

    frame = bytearray(19)
    frame[0] = 0xFA
    h, l = _u16_be(gsr_raw)
    frame[1], frame[2] = h, l

    for i, raw in enumerate(acc_raw_xyz):
        rh, rl = _u16_be(raw & 0xFFFF)
        frame[3 + i * 2], frame[4 + i * 2] = rh, rl

    for i, raw in enumerate(gyro_raw_xyz):
        rh, rl = _u16_be(raw & 0xFFFF)
        frame[9 + i * 2], frame[10 + i * 2] = rh, rl

    frame[15] = heart_rate & 0xFF
    frame[16] = spo2 & 0xFF

    chk = sum(frame[1:17]) % 256
    frame[17] = chk
    frame[18] = 0xAF
    return bytes(frame)
