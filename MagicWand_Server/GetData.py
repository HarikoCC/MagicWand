import asyncio
import os
import logging
from bleak import BleakScanner, BleakClient, BleakError

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 正确的 UUID（根据 Arduino 代码）
SERVICE_UUID = "0000fffa-0000-1000-8000-00805f9b34fb"
RECEIVE_UUID = "0000fffc-0000-1000-8000-00805f9b34fb"

# 存储目录（可修改）
SAVE_DIR = "./dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# 全局变量
receiving = False
buffer = []
current_filename = None  # 记录当前正在写入的文件名，用于中断时删除


def generate_filename():
    """生成下一个文件名，如 0.txt, 1.txt, ..."""
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".txt")]
    number = len(files)
    return os.path.join(SAVE_DIR, f"{number}.txt")


def parse_line(data: str) -> bool:
    """检查是否为有效的 24 个 '0'/'1' 组成的字符串"""
    return len(data) == 24 and all(c in '01' for c in data)


def cleanup_incomplete_file():
    """删除未完成的文件"""
    global current_filename
    if current_filename and os.path.exists(current_filename):
        os.remove(current_filename)
        logger.info(f"🗑️ Deleted incomplete file: {current_filename}")
        current_filename = None


async def handle_notification(sender, data: bytearray):
    global receiving, buffer, current_filename

    try:
        msg = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        logger.warning(f"Invalid UTF-8 data from {sender}: {data.hex()}")
        return

    print(f"Received: {msg}")

    if msg == "DATA_BEGIN":
        receiving = True
        buffer = []
        current_filename = generate_filename()  # 预分配文件名
        logger.info("Start receiving 24x24 grid data...")
    elif msg == "DATA_END":
        if receiving and len(buffer) == 24:
            try:
                with open(current_filename, "w") as f:
                    for line in buffer:
                        f.write(line + "\n")
                logger.info(f"✅ Data saved to {current_filename}")
            except Exception as e:
                logger.error(f"Failed to write file {current_filename}: {e}")
            finally:
                current_filename = None
        else:
            logger.warning(f"❌ Incomplete data received: {len(buffer)} lines (expected 24)")
            cleanup_incomplete_file()
        receiving = False
    elif receiving:
        if parse_line(msg):
            buffer.append(msg)
            if len(buffer) == 24:
                logger.info("Received 24 lines, waiting for DATA_END...")
        else:
            logger.warning(f"⚠️ Invalid line ignored: {msg}")
    else:
        # 收到非协议消息（不在接收状态）
        pass


async def listen_for_data(address):
    global receiving, buffer

    while True:
        client = BleakClient(address)
        try:
            logger.info(f"Connecting to {address}...")
            await client.connect()
            if not client.is_connected:
                logger.error("Failed to connect.")
                await asyncio.sleep(5)
                continue

            logger.info("Connected. Starting notification...")
            await client.start_notify(RECEIVE_UUID, handle_notification)

            # 持续运行，直到连接断开或被取消
            while client.is_connected:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Connection task was cancelled.")
            if client.is_connected:
                await client.stop_notify(RECEIVE_UUID)
                await client.disconnect()
            cleanup_incomplete_file()
            break

        except Exception as e:
            logger.error(f"Connection lost or error occurred: {type(e).__name__}: {e}")
            cleanup_incomplete_file()

        finally:
            if client.is_connected:
                try:
                    await client.stop_notify(RECEIVE_UUID)
                    await client.disconnect()
                except:
                    pass

        # 连接失败或断开后，等待后重新尝试连接
        receiving = False
        buffer = []
        await asyncio.sleep(3)  # 避免频繁重连


async def scan_and_connect_forever():
    """持续扫描并连接 MagicWand 设备"""
    while True:
        logger.info("Scanning for BLE devices...")
        try:
            devices = await BleakScanner.discover(timeout=10.0)  # 10秒扫描
        except Exception as e:
            logger.error(f"Scanner error: {e}")
            await asyncio.sleep(5)
            continue

        target_device = None
        for d in devices:
            if d.name == "MagicWand":
                target_device = d
                break

        if target_device:
            logger.info(f"Found MagicWand at {target_device.address}")
            try:
                # 运行监听任务，如果取消或出错会返回
                await listen_for_data(target_device.address)
            except Exception as e:
                logger.error(f"Error during data listening: {e}")
                cleanup_incomplete_file()
        else:
            logger.info("Device 'MagicWand' not found. Retrying...")
            await asyncio.sleep(2)  # 扫描间隔


async def main():
    await scan_and_connect_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Program interrupted by user.")
        # 确保中断时也能清理文件
        cleanup_incomplete_file()