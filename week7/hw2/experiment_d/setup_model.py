"""取得公開 Clef-Flash GGUF 決策模型，固定版本並驗證 LFS SHA-256。"""
from classifier.common import save_json
from hw2.experiment_a.prompt import ROOT
from hw2.experiment_a.setup_local import download

REPO='ggml-org/Clef-Flash-GGUF'
REVISION='c3d925454595d792d5333da8b35f3c09911f7568'
NAME='Clef-Flash-Q4_K_M.gguf'
SHA='eef2968c238a80d49d340430f22410ea3db9934ad1c6fdb233a503cbf9a9b54e'
SIZE=6732781824


def main():
    """只下載既有發布的決策模型；不使用 test 訓練，也不呼叫付費 API。"""
    path=ROOT/'models/clef-flash-gguf'/NAME
    url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}'
    download(url,path,SHA,SIZE)
    save_json(path.parent/'source_manifest.json',{'model':'Cloudflare/clef-flash','gguf_repository':REPO,
        'revision':REVISION,'filename':NAME,'quantization':'Q4_K_M','parameter_count':'9B',
        'url':url,'sha256':SHA,'bytes':SIZE,'license':'Apache-2.0',
        'api':'/v1/systemone','native_decision_model':True,'assignment_fine_tuning':False,
        'source_model_card':'https://huggingface.co/Cloudflare/clef-flash',
        'runtime_model_card':'https://huggingface.co/ggml-org/Clef-Flash-GGUF'})
    print('Clef-Flash prepared:',path,flush=True)


if __name__=='__main__':
    main()
