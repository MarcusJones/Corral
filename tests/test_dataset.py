from corral.dataset import AIDataSet

def test_register_consume():
    ocean = OceanContracts(host='http://0.0.0.0', port=8545)
    resouce_id = register(publisher_account=ocean.web3.eth.accounts[1],
                          provider_account=ocean.web3.eth.accounts[0],
                          price=10,
                          ocean_contracts_wrapper=ocean,
                          json_metadata=json_consume,
                          provider_host='http://0.0.0.0:5000'
                          )
    assert requests.get('http://0.0.0.0:5000/api/v1/provider/assets/metadata/%s' % resouce_id).status_code == 200
