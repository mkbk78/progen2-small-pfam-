import sys, errno, re, json, ssl
import os
from urllib import request
from urllib.error import HTTPError
from time import sleep
import argparse
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_pfam_entry(base_url: str, download_file: str):
    # 禁用SSL验证以避免配置问题
    context = ssl._create_unverified_context()
    HEADER_SEPARATOR = "|"
    LINE_LENGTH = 80

    next = base_url# 将next设置为base_url，以便从API的第一个页面开始获取数据

    with open(download_file, "w") as f:

        attempts = 0# 初始化尝试次数为0
        while next:# 当next不为空时，继续循环
            try:
                req = request.Request(next, headers={"Accept": "application/json"})
                # 创建一个请求，指定URL和Accept头，以获取JSON数据。
                res = request.urlopen(req, context=context)# 使用urlopen打开请求，并读取响应
                # 如果API超时，可能是由于查询运行时间过长
                if res.status == 408:
                    # 等待超过一分钟
                    sleep(61)
                    # 然后继续此循环使用相同的URL
                    continue
                elif res.status == 204:
                    # 没有数据，跳出
                    break
                payload = json.loads(res.read().decode())
                next = payload["next"]
                attempts = 0 # 将尝试次数重置为0
            except HTTPError as e:
                if e.code == 408:
                    sleep(61)
                    continue
                else:
                    if attempts < 3:# 如果出现其他HTTP错误，将重试3次后失败
                        attempts += 1
                        sleep(61)
                        continue
                    else:# 如果尝试次数达到3次，仍然无法获取数据，则抛出异常。
                        sys.stderr.write("LAST URL: " + next)
                        raise e

            for i, item in enumerate(payload["results"]):# 如果尝试次数达到3次，仍然无法获取数据，则抛出异常。

                entries = None
                # 如果item中存在"entry_subset"键，则将对应的值赋给entries变量。
                if "entry_subset" in item:
                    entries = item["entry_subset"]
                # 如果item中存在"entries"键，则将对应的值赋给entries变量。
                elif "entries" in item:
                    entries = item["entries"]

                if entries is not None:
                    entries_header = "-".join(
                        [
                            entry["accession"]
                            + "("
                            + ";".join(
                                [
                                    ",".join(
                                        [
                                            str(fragment["start"])
                                            + "..."
                                            + str(fragment["end"])
                                            for fragment in locations["fragments"]
                                        ]
                                    )
                                    for locations in entry["entry_protein_locations"]
                                ]
                            )
                            + ")"
                            for entry in entries
                        ]
                    )
                    f.write(# 将entries_header、item中的"metadata"字典的"accession"和"name"键对应的值写入文件。
                        ">"
                        + item["metadata"]["accession"]
                        + HEADER_SEPARATOR
                        + entries_header
                        + HEADER_SEPARATOR
                        + item["metadata"]["name"]
                        + "\n"
                    )
                else:
                    f.write(# 将item中的"extra_fields"字典的"sequence"键对应的值写入文件
                        ">"
                        + item["metadata"]["accession"]
                        + HEADER_SEPARATOR
                        + item["metadata"]["name"]
                        + "\n"
                    )
                # 将item中的"extra_fields"字典的"sequence"键对应的值写入文件
                seq = item["extra_fields"]["sequence"]
                fastaSeqFragments = [
                    seq[0 + i : LINE_LENGTH + i]
                    for i in range(0, len(seq), LINE_LENGTH)
                ]
                for fastaSeqFragment in fastaSeqFragments:
                    f.write(fastaSeqFragment + "\n")


            if next:# 如果next不为空，等待一秒钟再继续。
                sleep(1)


def main(args):
    base_url = "https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/pfam/{}/?page_size=200&extra_fields=sequence"
    #pfam的url
    os.makedirs("downloads", exist_ok=True)#创建文件夹存放下载fasta文件

    for pfam_code in args.pfam_codes:# 将entries_header、item中的"metadata"字典的"accession"和"name"键对应的值写入文件。
        if not re.match(r"PF[0-9]{5}", pfam_code):# 如果Pfam代码不符合格式要求（必须以"PF"开头，后跟5个数字），则抛出异常。
            raise Exception(f'Pfam code not valid. Must be "PF" followed by 5 digits, got: {pfam_code}. Example: PF12345')
        download_file = os.path.join("downloads", f"{pfam_code}.fasta")# 创建一个名为download_file的字符串，其中包含Pfam代码对应的下载文件名。
        url = base_url.format(pfam_code)# 创建一个名为url的字符串，其中包含根据Pfam代码生成的API URL。
        logger.info(f"Downloading {pfam_code} from {url}")# 打印一条日志，表示正在下载Pfam代码对应的蛋白质序列。
        download_pfam_entry(url, download_file)# 调用download_pfam_entry函数，下载Pfam代码对应的蛋白质序列并将其保存到文件中。
        logger.info(f"Downloaded {pfam_code} to {download_file}")# 打印一条日志，表示已经下载了Pfam代码对应的蛋白质序列并将其保存到文件中。


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pfam_codes",
        nargs="+",
        help='pfam——codes格式要满足“PF”+五位数字. Example: PF12345',
    )
    args = parser.parse_args()
    main(args)
