import requests
import json
import sys
from pathlib import Path
import pkg_resources
import docutils

PARENT_ID = 698342
BASE_URL = 'https://sandbox.zenodo.org/api/deposit/depositions/'

def publish(path_dist, access_token, verbose=False):
    """Publish a new version of software to Zenodo

    Parameters:
    access_token (str): Zenodo access token (https://developers.zenodo.org/)
    path_dist (str): Path to folder containing distributions
    verbose (bool): Print HTTP status codes

    """

    params = {'access_token': access_token}
    headers = {"Content-Type": "application/json"}

    # Create a new version of the deposition
    r = requests.post(BASE_URL + '{}/actions/newversion'.format(PARENT_ID),
                        params=params,
                        json={},
                        headers=headers)
    
    if verbose:
        print('Create a new version of the deposition: {}'.format(r.status_code))

    # Get the new version, its id and bucket_url
    r = requests.get(r.json()['links']['latest_draft'], params=params)
    deposition_id = r.json()['id']
    bucket_url = r.json()["links"]["bucket"]

    if verbose:
        print('Get the new version: {}'.format(r.status_code))
        print('id: {}'.format(deposition_id))
        print('bucket_url: {}'.format(bucket_url))

    # Delete existing files
    for file in r.json()['files']:
        requests.delete(BASE_URL + '%s/files/%s' % (deposition_id, file['id']), params=params)

    # Locate distributuon file
    files = [file for file in Path(path_dist).glob('**/*') if file.name.endswith('tar.gz')]

    # Put distribution file
    with files[0].open(mode="rb") as fp:
        r = requests.put(
            '{}/{}'.format(bucket_url, files[0].name),
            data=fp,
            params=params,
        )

    if verbose:
        print('Put distribution file: {}'.format(r.status_code))

    # Load metadata
    metadata = load_metadata()

    # Put metadata
    r = requests.put(BASE_URL + '%s' % deposition_id, params=params, data=json.dumps(metadata), headers=headers)

    if verbose:
        print('Put metadata: {}'.format(r.status_code))

    # Publish new version
    r = requests.post(BASE_URL + '%s/actions/publish' % deposition_id, params=params )

    if verbose:
        print('Publish new version: {}'.format(r.status_code))

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
    
