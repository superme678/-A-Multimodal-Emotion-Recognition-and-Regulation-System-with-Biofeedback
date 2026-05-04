import time


def parse_valid_frames(buffer):
    """解析完整数据帧，返回结果+剩余残帧缓存"""
    valid_frames = []
    FIXED_FRAME_LEN = 19
    FRAME_HEAD = 0xFA
    FRAME_TAIL = 0xAF

    i = 0
    total_len = len(buffer)
    while i <= total_len - FIXED_FRAME_LEN:
        if buffer[i] != FRAME_HEAD:
            i += 1
            continue

        frame = buffer[i:i + FIXED_FRAME_LEN]
        if frame[-1] != FRAME_TAIL:
            i += 1
            continue

        # 校验和验证
        data_sum = sum(frame[1:17])
        calc_checksum = data_sum % 256
        real_checksum = frame[17]
        if calc_checksum != real_checksum:
            i += 1
            continue

        # GSR计算
        gsr_raw = frame[1] * 256 + frame[2]
        gsr_voltage = round((gsr_raw / 4095) * 3.3, 4)

        # 加速度计算
        def calc_acc(high, low):
            raw = high * 256 + low
            signed = raw if raw < 32768 else raw - 65536
            return round((signed / 32768) * 2, 4)

        acc_x = calc_acc(frame[3], frame[4])
        acc_y = calc_acc(frame[5], frame[6])
        acc_z = calc_acc(frame[7], frame[8])

        # 陀螺仪计算
        def calc_gyro(high, low):
            raw = high * 256 + low
            signed = raw if raw < 32768 else raw - 65536
            return round((signed / 32768) * 250, 4)

        gyro_x = calc_gyro(frame[9], frame[10])
        gyro_y = calc_gyro(frame[11], frame[12])
        gyro_z = calc_gyro(frame[13], frame[14])

        # 心率、血氧
        heart_rate = frame[15]
        spo2 = frame[16]

        valid_frames.append([gsr_raw, gsr_voltage, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, heart_rate, spo2])
        i += FIXED_FRAME_LEN

    return valid_frames, buffer[i:]


def real_time_logger(dat_file_path, save_txt_path):
    # 全局变量
    cache_buffer = b''
    packet_index = 0

    # 打印表头
    header = "=" * 120 + "\n序号 | GSR(原始值/电压) | 加速度X(g) | 加速度Y(g) | 加速度Z(g) | 陀螺仪X(°/s) | 陀螺仪Y(°/s) | 陀螺仪Z(°/s) | 心率 | 血氧\n" + "=" * 120
    print(header)

    # 初始化TXT文件（写入表头）
    with open(save_txt_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")

    try:
        while True:
            new_data = b''
            # 1. 读取DAT文件所有数据
            with open(dat_file_path, 'rb') as f:
                new_data = f.read()

            if new_data:
                # 2. 拼接残帧 + 新数据
                cache_buffer += new_data
                # 3. 解析完整有效帧
                valid_frames, cache_buffer = parse_valid_frames(cache_buffer)

                # 4. 实时输出 + 实时写入TXT
                for data in valid_frames:
                    packet_index += 1
                    gsr_raw, gsr_volt, acc_x, acc_y, acc_z, gx, gy, gz, hr, spo2 = data
                    # 控制台单行输出
                    line = f"{packet_index:>2} | {gsr_raw}({gsr_volt}V) | {acc_x:>9} | {acc_y:>9} | {acc_z:>9} | {gx:>10} | {gy:>10} | {gz:>10} | {hr:>2}bpm | {spo2:>2}%"
                    print(line)
                    # TXT文件实时追加写入
                    with open(save_txt_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")

                # 5. 核心：清空DAT文件，防止冗余数据
                with open(dat_file_path, 'wb') as f:
                    f.truncate(0)
                    f.seek(0)

            # 高频监听
            time.sleep(0.01)

    except KeyboardInterrupt:
        # ===================== 程序停止时：清空TXT文件 =====================
        with open(save_txt_path, 'wb') as f:
            f.truncate(0)  # 清空TXT所有数据
        # ==================================================================

        print(f"\n✅ 实时解析已停止！")
        print(f"📊 累计解析有效数据包：{packet_index}")
        print(f"🗑️ 已自动清空【TXT结果文件】和【DAT原始文件】")
        print(f"📂 TXT文件路径：{save_txt_path}")


if __name__ == '__main__':
    # ===================== 仅修改这两个路径 =====================
    DAT_FILE = r"D:\AI_project\Facial-and-Speech-Dual-Modal-Emotion-Sensing-and-Regulation-System-master\Facial_Voice_Sense\sscom\ReceivedTofile-COM3-2026_5_3_14-28-49.DAT"  # 串口实时写入的文件
    SAVE_TXT = r"D:\AI_project\\Facial-and-Speech-Dual-Modal-Emotion-Sensing-and-Regulation-System-master\Facial_Voice_Sense\sscom\手环实时解析结果.txt" # 计算结果保存的TXT
    # ============================================================

    real_time_logger(DAT_FILE, SAVE_TXT)