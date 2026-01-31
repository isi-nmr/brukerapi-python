import json
import sys
from pathlib import Path

import pkg_resources
import requests

PARENT_ID = 698342
BASE_URL = 'https://sandbox.zenodo.org/api/deposit/depositions/'

def publish(path_dist, access_token,*, verbose=False):
    """Publish a new version of software to Zenodo

    Parameters:
    access_token (str): Zenodo access token (https://developers.zenodo.org/)
    path_dist (str): Path to folder containing distributions
    verbose (bool): Print HTTP status codes

    """

    params = {'access_token': access_token}
    headers = {"Content-Type": "application/json"}

    # Create a new version of the deposition
    r = requests.post(BASE_URL + f'{PARENT_ID}/actions/newversion',
                        params=params,
                        json={},
                        headers=headers)

    if verbose:
        print(f'Create a new version of the deposition: {r.status_code}')

    # Get the new version, its id and bucket_url
    r = requests.get(r.json()['links']['latest_draft'], params=params)
    deposition_id = r.json()['id']
    bucket_url = r.json()["links"]["bucket"]

    if verbose:
        print(f'Get the new version: {r.status_code}')
        print(f'id: {deposition_id}')
        print(f'bucket_url: {bucket_url}')

    # Delete existing files
    for file in r.json()['files']:
        requests.delete(BASE_URL + '{}/files/{}'.format(deposition_id, file['id']), params=params)

    # Locate distributuon file
    files = [file for file in Path(path_dist).glob('**/*') if file.name.endswith('tar.gz')]

    # Put distribution file
    with files[0].open(mode="rb") as fp:
        r = requests.put(
            f'{bucket_url}/{files[0].name}',
            data=fp,
            params=params,
        )

    if verbose:
        print(f'Put distribution file: {r.status_code}')

    # Load metadata
    metadata = load_metadata()

    # Put metadata
    r = requests.put(BASE_URL + f'{deposition_id}', params=params, data=json.dumps(metadata), headers=headers)

    if verbose:
        print(f'Put metadata: {r.status_code}')

    # Publish new version
    r = requests.post(BASE_URL + f'{deposition_id}/actions/publish', params=params )

    if verbose:
        print(f'Publish new version: {r.status_code}')

def get_version():
    return pkg_resources.get_distribution("brukerapi").version

def load_metadata():
    with open(Path(__file__).parent / 'fixed.json') as f:
        data = json.load(f)

    data['metadata']['version'] = get_version()

    return data

def append_changelog():
    pass

if __name__ == "__main__":
    publish(sys.argv[0], sys.argv[1])

