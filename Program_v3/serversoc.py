# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# 接收端
import socket
import os
import time
from struct import pack
from struct import unpack
from tqdm import tqdm
import zipfile
import sys

# DOWNLOAD_PATH = R'C:\Users\Archillesheel\Desktop\python文件传输'  # 传输目录
# DOWNLOAD_PATH = R'D:\PycharmProjects\VE490\Receive'
DOWNLOAD_PATH = '.\Receive'


def get_host_ip():
    """
    查询本机ip地址
    :return:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        print("本机局域网ip地址：", ip)
    finally:
        s.close()
    return ip


def unzip_file(zip_src, zip_dir):
    if zipfile.is_zipfile(zip_src):
        zf = zipfile.ZipFile(zip_src, 'r')
        for each in zf.namelist():
            zf.extract(each, zip_dir)
    else:
        print("It's not zip!")


def send_file(file_name, file_socket: socket.socket):
    try:
        f = open(file_name, 'rb')
        size = os.path.getsize(file_name)
        if size < 1024:
            read_size = 1024
        elif size < 1024 * 1024 and size >= 1024:
            read_size = 1024 * 1024
        else:
            read_size = 500 * 1024 * 1024
        file_socket.send(pack('q', size))
        del size
        file_socket.recv(1024)
        while True:
            data = f.read(read_size)
            if not data:
                break
            file_socket.send(data)
        f.close()
    except FileNotFoundError:
        print(f'没有找到{file_name}')


def download(file_name, file_socket: socket.socket):
    file_size = unpack('q', file_socket.recv(1024))[0]
    if file_size < 1024:
        print(f'文件大小：{file_size} B')
    elif file_size < 1024 * 1024 and file_size >= 1024:
        print(f'文件大小：{file_size / 1024} KB')
    else:
        print(f'文件大小：{file_size / (1024 * 1024)} MB')
    f = open(file_name, 'wb')
    print('开始传输...')
    download_size = 2048
    file_socket.send('开始传输'.encode('gbk'))
    start = time.time()
    for i in tqdm(range(int(file_size / download_size) + 1)):
        data = file_socket.recv(download_size)
        f.write(data)
    end = time.time()
    f.close()
    print('传输完成！')
    print(f'大约耗时{end - start}秒')


if __name__ == '__main__':
    # 判断是否有ip地址配置文件。如果没有添加文件ipconfig.xml，录入ip地址；如果有，检查配置文件中ip地址是否与当前相同。
    judge = False
    for files, dirs, root in os.walk(os.path.abspath('.')):
        for file in files:
            if "ipconfig.xml" in file:
                judge = True
                print("ip文件已配置！")
                break
    if not judge:
        f = open("ipconfig.xml", "w+")
        f.writelines(get_host_ip())
        print("ip地址已写入！")
        f.close()
    else:
        f = open("ipconfig.xml")
        line = f.readline().split("\n")[0]
        if line != get_host_ip():
            f.truncate()
            f.writelines(get_host_ip())
            print("配置文件更新，请更改client端配置文件！")
        f.close()
    # 开放端口，建立socket，进行一个包的接受
    os.chdir(DOWNLOAD_PATH)  # DOWNLOAD_PATH是一个全局变量，表示存放图片的地址
    while True:
        try:
            file_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            file_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
            file_socket_port = 8080
            # file_socket_port = int(input('请输入端口号：'))
            ip_local = socket.gethostbyname(socket.gethostname())
            file_socket.bind((ip_local, file_socket_port))
            print('成功启动，等待连接。。。')
            file_socket.listen(128)
            f_socket, f_addr = file_socket.accept()
            print(f'建立连接{f_addr}')
            f_socket.send('请输入文件名'.encode('gbk'))
            file_name = f_socket.recv(1024)
            download(file_name.decode('gbk'), f_socket)
            f_socket.close()
            file_socket.close()
            time.sleep(3)
            try:
                file_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                file_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
                ip = f_addr[0]
                port = 8080
                file_socket.connect((ip, port))
                print('连接成功，准备开始传输。。。')
                path = os.path.join(sys.path[0], "Receive")
                file_socket.recv(1024).decode('gbk')
                # 压缩包路径及名字
                result_name = "coord.txt"
                start_flag = False
                while not start_flag:
                    for root, dirs, files in os.walk(path):
                        if result_name in files:
                            start_flag = True
                            print("file exist")
                        else:
                            time.sleep(3)
                            print("waiting for file...")
                time.sleep(1)
                file_socket.send(result_name.encode('gbk'))
                result_file = os.path.join(sys.path[0], "Receive", result_name)
                send_file(result_file, file_socket)
            except ConnectionResetError:
                print('接收端断开连接')
        except ConnectionResetError:
            print('发送端已断开连接')
        except UnicodeDecodeError:
            print('文件编码错误，请检查文件格式是否正确')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
