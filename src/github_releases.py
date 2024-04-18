import requests
from langchain.tools import tool   


def get_headers(token: str):
    headers = {
        'Accept': 'application/vnd.github+json',
        'Authorization': 'Bearer '+token,
        'X-GitHub-Api-Version': '2022-11-28'
        }
    return headers


def fetch_github_tags(owner='greenqloud', repo='nkdev', max_pages=5, token=None):
    headers = get_headers(token)
    page=1
    items = []
    for page in range(1, max_pages):
        print('Requesting page: ', page)
        x = requests.get('https://api.github.com/repos/{owner}/{repo}/tags?page={page}'.format(owner=owner, repo=repo, page=page), 
                        headers=headers)
        if x.status_code != 200:
            print('Error in request, code: ', x.status_code)
            break 
        page=page+1
        newitems = x.json()
        if len(newitems) == 0:
            break
        items = items+ newitems
    return items

def fetch_github_latest(owner='greenqloud', repo='nkdev', token=None):
    headers = get_headers(token)
    x = requests.get('https://api.github.com/repos/{owner}/{repo}/releases/latest'.format(owner=owner, repo=repo), 
                    headers=headers)
    
    
    if x.status_code != 200:
        print('Error in request, code: ', x.status_code)
        return 
    
    return x.json()
    #reqbody = x.json()
    
    #return reqbody['body']    

def fetch_github_tag(owner='greenqloud', repo='nkdev', token=None, tag_name='v1.0.0'):
    headers = get_headers(token)
    request_str = 'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag_name}'.format(owner=owner, repo=repo, tag_name=tag_name)
    print('Requesting: ', request_str)
    x = requests.get(request_str, headers=headers)
    if x.status_code != 200:
        print('Error in request, code: ', x.status_code)
        return 
    
    return x.json()
    #reqbody = x.json()
    #print(reqbody['body'])
    #return reqbody['body']
        
def get_github_pat():
    import os
    pat = os.getenv("PAT_GITHUB")
    if pat == "":
        print("No PAT_GITHUB env variable found")
    return pat

def fix_tag(tag):
    if tag.startswith('v'):
        return tag
    return 'v'+tag
        
@tool
def github_version(tag: str, repo:str, owner='greenqloud'):
    """Gets a certain version or a tag of a repository. Versions should be in semver format (e.g. v1.2.3)"""
    pat = get_github_pat()
    taginfo = fetch_github_tag(repo=repo, tag_name=fix_tag(tag), token=pat, owner=owner)
    if taginfo is not None and 'body' in taginfo:
        return taginfo
        #return taginfo['body']
    print('Tag not found or no body')
    
    return None

@tool
def github_latest_release(repo:str, owner='greenqloud'):
    """Gets the latest release of a repository"""
    pat = get_github_pat()
    return fetch_github_latest(repo=repo, token=pat, owner=owner)


@tool
def github_get_tags(repo:str, owner='greenqloud'):
    """Gets lost of the tags and releases of a repository"""
    pat = get_github_pat()
    return fetch_github_tags(repo=repo, token=pat, owner=owner)

def get_toolkit():
    return [github_version, github_latest_release, github_get_tags]
#github_version(tag='1.8.1')