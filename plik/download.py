import requests
import tarfile

for filename in ["plikHM_TTTEEE_lowl_lowE_lensing.tar.gz","plikHM_TTTEEE_lowl_lowE_lensing_NS.tar.gz"]:
    github_url = "https://github.com/williamjameshandley/cosmo_example/raw/master/"
    url = github_url + filename
    open(filename, 'wb').write(requests.get(url).content)
    tarfile.open(filename).extractall()

from anesthetic import NestedSamples
nested_root = 'plikHM_TTTEEE_lowl_lowE_lensing_NS/NS_plikHM_TTTEEE_lowl_lowE_lensing'
nested = NestedSamples(root=nested_root)
