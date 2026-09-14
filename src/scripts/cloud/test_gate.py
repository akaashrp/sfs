"""Bind a successful, non-skipped regression run to unchanged runtime source."""
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.cloud.common import source_hashes,read,write,digest

if __name__=='__main__':
    mode,folder=sys.argv[1],Path(sys.argv[2])
    if mode=='start':write(folder/'source_before.json',source_hashes())
    else:
        source=source_hashes()
        if read(folder/'source_before.json')!=source:raise ValueError('Source changed during regression tests')
        suites=list(ET.parse(folder/'results.xml').getroot().iter('testsuite'))
        tests=sum(int(s.get('tests',0)) for s in suites)
        if tests<70 or any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped')):
            raise ValueError('Regression tests failed, skipped, or incomplete')
        write(folder/'gate.json',{'status':'PASS_CPU_REGRESSION','tests':tests,'source_sha256':source,
                                 'xml_sha256':digest(folder/'results.xml')})
