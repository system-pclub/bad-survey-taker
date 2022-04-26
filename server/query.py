"""need sudo to run"""

import os, json
from datetime import datetime
import requests, math
from pprint import pprint
import vt, minfraud, jsons, pydnsbl
import sqlite3, subprocess

db_path = r"./data/query.db"
keys = {}


def get_utcnow_second():
    return math.floor(datetime.now().timestamp())

def initialize_db(db_path=db_path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            '''CREATE TABLE if not exists Queries
                    (IP TEXT PRIMARY KEY, start_utc INTEGER, complete_utc INTEGER, VirusTotal TEXT, IPQualityscore TEXT, MinFraud TEXT, IPRegistry TEXT, DNSBL TEXT, nmap TEXT)
        ''')
        conn.commit()


class BaseRequester:
    def make_request(self, ip):
        return ""


class VTRequest():
    def __init__(self, api_key: str = keys.get("VT")):
        self.client = vt.Client(api_key)

    def make_request(self, ip):
        response = self.client.get_object(f"/ip_addresses/{ip}").to_dict()
        if "last_analysis_stats" not in response["attributes"]:
            raise Exception(f"{response}")
        return response


class IPQualityscoreRequest():

    def __init__(self, api_key: str = keys.get("IPQuality")):
        self.api_key = api_key

    def make_request(self, ip: str):
        base_url = f"https://ipqualityscore.com/api/json/ip/{self.api_key}/{ip}?strictness=0&allow_public_access_points=true&fast=true&lighter_penalties=true&mobile=true"
        response = requests.get(base_url)
        response = response.json()
        if not response["success"]:
            raise Exception(f"{response}")
        return response


class MaxmindMinfraudRequest():

    def __init__(self, account_id: int = 543647, api_key: str = keys.get("maxmind")):
        self.client = minfraud.Client(account_id, api_key)

    def make_request(self, ip: str):
        response = self.client.insights({'device': {'ip_address': ip}})
        response = jsons.dump(response)
        return response


class IpregistryRequest():

    def __init__(self, api_key: str = keys.get("IPregistry")):
        self.http_queries = {"key": api_key}

    def make_request(self, ip):
        MAX_ENTRIES = 255

        if type(ip) == str:
            s = ip
            base_url = f"https://api.ipregistry.co/{s}"
            response = requests.get(base_url, params=self.http_queries)
            response.raise_for_status()
            return response.json()
        elif type(ip) == list:
            ips = [ip[i*MAX_ENTRIES: (i+1)*MAX_ENTRIES] for i in range(math.ceil(len(ip)/MAX_ENTRIES))]
            results = []
            for ip_batch in ips:
                if len(ip_batch) > 0:
                    s = ",".join(ip_batch)
                    base_url = f"https://api.ipregistry.co/{s}"
                    response = requests.get(base_url, params=self.http_queries)
                    response.raise_for_status()
                    if len(ip_batch) == 1:
                        results.append(response.json())
                    else:
                        results.extend(response.json()["results"])
            return results
        else:
            raise TypeError(f"Wrong type of `ip`.\n Expecting `str|List[str]`, found `{type(ip)}`\n")


class DNSBLRequest:

    def __init__(self):
        self.checker = pydnsbl.DNSBLIpChecker()

    def make_request(self, ip: str):
        result = self.checker.check(ip)
        result = jsons.dump(result)
        del result["_results"]
        return result


nmap_nports_list = [10, 50, 100, 500, 1000]
class NmapProbe:

    def __init__(self):
        self

    def make_single_request(self, ip, nports=1000):
        from subprocess import PIPE
        cmd = f"sudo nmap -sTU --top-ports {nports} -O {ip}"
        proc = subprocess.run(cmd, shell=True, stderr=PIPE, stdout=PIPE)
        out = proc.stderr.decode('utf-8') + "\n" + proc.stdout.decode('utf-8')
        result = {"top-ports":nports, "cmd": cmd, "output": out,}
        return result

    def make_request(self, ip, nports_list=nmap_nports_list):
        from subprocess import PIPE
        results = []
        for nports in nports_list:
            start_utc = get_utcnow_second()
            result = self.make_single_request(ip, nports)
            end_utc = get_utcnow_second()
            result["start_utc"] = start_utc
            result["end_utc"] = end_utc
            results.append(result)
        return results
        

class Requester:

    def __init__(self, db_path=db_path):
        self.requesters = {
            "VirusTotal": VTRequest(),
            "IPQualityscore": IPQualityscoreRequest(),
            "MinFraud": MaxmindMinfraudRequest(),
            "IPRegistry": IpregistryRequest(),
            "DNSBL": DNSBLRequest(),
            # "nmap": NmapProbe(),
            "nmap": BaseRequester(),
        }
        self.db_path = db_path
        initialize_db(self.db_path)

    def _make_request(self, ip: str):
        results = {}
        for key, requester in self.requesters.items():
            success = True
            start_utc = None
            end_utc = None
            try:
                start_utc = get_utcnow_second()
                response = requester.make_request(ip)
                end_utc = get_utcnow_second()
            except Exception as e:
                response = {"output": str(e)}
                success = False
            query_utc = get_utcnow_second()
            result = {
                "success": success,
                "response": response,
                "start_utc": start_utc,
                "end_utc": end_utc
            }
            results[key] = json.dumps(result)
        return results
        
    def request_store(self, ip):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            start_utc = get_utcnow_second()
            try:
                cur.execute('''
                INSERT INTO Queries (IP, start_utc)
                VALUES (?, ?)''',
                (ip, start_utc))
            except sqlite3.IntegrityError as e:
                return
            results = self._make_request(ip)
            complete_utc = get_utcnow_second()
            cur = conn.cursor()
            cur.execute('''
                UPDATE Queries 
                SET complete_utc=?, 
                    VirusTotal=?,
                    IPQualityscore=?,
                    MinFraud=?,
                    IPRegistry=?,
                    DNSBL=?,
                    nmap=?
                WHERE ip=?
                ''',
                (complete_utc, results["VirusTotal"], results["IPQualityscore"], results["MinFraud"], results["IPRegistry"], results["DNSBL"], results["nmap"], ip)
            )
            conn.commit()
            

if __name__ == "__main__":
    import sys
    ip = sys.argv[-1]
    requester = Requester()
    requester.request_store(ip)


