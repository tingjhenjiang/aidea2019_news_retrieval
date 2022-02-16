from __future__ import unicode_literals
from flask import Flask, request, render_template, abort, jsonify, make_response, Response, json
from flask_restful import Resource, Api
import importlib,newsgetdata,constructmodel,newsrecommend
import tensorflow as tf
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import os,sys

app = Flask(__name__,template_folder='templates')
app.config['JSON_AS_ASCII'] = False
api = Api(app)
instanceGetdata = newsgetdata.getdata()
instanceConstructmodel = constructmodel.constructmodel(
    ckipModelName='ckiplab/albert-tiny-chinese'
)
instanceRecommend = newsrecommend.recommend()

#if os.environ.get('WERKZEUG_RUN_MAIN') != 'true': #Only run once inside this block; using flask run
needData = instanceGetdata.readsrc(
    candidate_negative_sampling_rate=0.01
    )

searchResInputPairedDf = needData['checkTokensDF']['Doc'].reset_index() #'News_URL','News_FullTitleContent'
searchResInputPairedDf = searchResInputPairedDf.merge(needData['compset'], how='left', on='News_URL')
#searchResInputPairedDf.to_csv('test.csv')
#display('complete writing to csv')
#'News_URL', 'News_FullTitleContent', 'Query', 'News_Index', 'Relevance', 'Query_Index', 'sampleweight', 'positionInQueryUniquedTokens', 'positionInDocUniquedTokens'
searchResInputPairedDf = searchResInputPairedDf.drop_duplicates(subset=['News_URL']).set_index('index')

models = instanceConstructmodel.generateTwoTowerModels(
    modelWeightsFile=os.path.join(
    os.getcwd(),
    'ckpt',
    'condors_twotower_2x256-tinyalbertbase-lrsched-sampleweightAndAugSamples-1.3125-elu_adam0.000217_dropout0.1085',
    'epoch364-loss66.519-val_loss68.854-weights.hdf5'
))
docEmbeddings = instanceGetdata.extractEmbedding(models['docModel'], needData['UniquedTokens']['Doc'])
faissIndex = instanceRecommend.generateRecommendModel(docEmbeddings=docEmbeddings['normalized'])


#if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

class apiresource_recommend(Resource):

    def __init__(self, **settings):
        super(apiresource_recommend, self).__init__()
        with open('qualified_token.txt') as f:
            qualified_token = f.readlines()
        settings.setdefault('batch_size', 32)
        settings.setdefault('qualified_token', qualified_token[0])
        self.settings = settings
        self.docEmbeddings = docEmbeddings
        self.faissIndex = faissIndex

    def get(self):
        query = request.args.get('query', None)
        if isinstance(query,str):
            query = query.strip()
        onestage = request.args.get('onestage', False)
        showtrainingdata = request.args.get('showtrainingdata', False)
        self.req_token = request.args.get('token', None)
        inhtml = request.args.get('inhtml', False)
        displayTrainingDataSampleN = int(request.args.get('displayTrainingDataSampleN', 1000))
        if showtrainingdata:
            sampleObserves = needData['compset'].index.to_series().sample(n=displayTrainingDataSampleN).to_numpy()
            modelInputQueryEmbeddings = instanceGetdata.extractEmbedding(models['queryModel'], modelInputs=needData['UniquedTokens']['Query'], prefix='Query_')
            modelInputQueryEmbeddings = modelInputQueryEmbeddings['notnormalized'][ needData['compset'].loc[sampleObserves,'positionInQueryUniquedTokens'].to_numpy() ]
            modelInputDocEmbeddings = self.docEmbeddings['notnormalized'][ needData['compset'].loc[sampleObserves,'positionInDocUniquedTokens'].to_numpy() ]
            print(f'modelInputQueryEmbeddings shape is {modelInputQueryEmbeddings.shape} and modelInputDocEmbeddings shape is {modelInputDocEmbeddings.shape}')
            rankingRes = self.ranking(
                queryEmbedding=modelInputQueryEmbeddings,
                docEmbeddings=modelInputDocEmbeddings,
                rankingModel=models['rankingModel'],
                pairedDF=needData['compset'].loc[sampleObserves].merge(needData['newsdf'], on='News_URL', how='left').drop(columns=['News_Fullcontent','News_FullTitleContent_fullStrlen'])
                )
            """
            relatedQueries = [query]+instanceGetdata.settings['querySynonyms'][query]+instanceGetdata.settings['queryAntonyms'][query]
            relatedData = needData['compset'][needData['compset']['Query'].isin(relatedQueries)].drop(columns=['Query_Index','News_URL','News_Index','sampleweight'])
            """
        else:
            queryTokens = instanceGetdata.getCKIPTokens(query)
            queryEmbedding = instanceGetdata.extractEmbedding(models['queryModel'], modelInputs=queryTokens, prefix='Query_')
            
            #retrieval depends on whether one stage method applies
            if onestage:
                retrievalFuncRes = {
                    #'queryStr': query,
                    'searchRes': np.reshape(np.arange( self.docEmbeddings['notnormalized'].shape[0] ), (1,-1)  ), #(n_query, topk)
                    'searchResDf': searchResInputPairedDf,#a paired search result dataframe
                }
            else:
                retrievalFuncRes = self.retrieval(queryEmbedding['normalized'], pairedDF=searchResInputPairedDf)

            rankingRes = self.ranking(
                queryEmbedding=queryEmbedding['notnormalized'],
                docEmbeddings=self.docEmbeddings['notnormalized'][retrievalFuncRes['searchRes'][0]],
                rankingModel=models['rankingModel'],
                pairedDF=retrievalFuncRes['searchResDf']
            )
        rankingRes = rankingRes.drop(columns=['Query_Index','News_URL','News_Index','sampleweight','positionInQueryUniquedTokens','positionInDocUniquedTokens'])
        probColumns = sorted([col for col in rankingRes.columns if col.find('prob')!=-1], reverse=True)
        for col in probColumns:
        #    rankingRes[col] = rankingRes[col].round(decimals=3).astype('float16')
            rankingRes[col] = rankingRes[col].apply(lambda x: round(x,3))#.astype('float16')
        rankingRes = rankingRes.sort_values(by=probColumns, ascending=False)
        displayRes = rankingRes

        if inhtml!=False:
            headers = {'Content-Type': 'text/html'}
            return make_response(render_template('restable.html', query=query, column_names=displayRes.columns.values, row_data=list(displayRes.values.tolist()), zip=zip), 200, headers)
        else:
            json_string = json.dumps(displayRes.to_dict(), ensure_ascii = False)
            response = Response(json_string,content_type="application/json; charset=utf-8" )
            return response

    def retrieval(self, queryEmbedding, **retrievalKwargs):
        retrievalKwargs.setdefault('pairedDF', searchResInputPairedDf)
        retrievalKwargs.setdefault('topk', 200)
        
        """
        returns: dict(
                'queryEmbedding': input
                'searchRes' : result matrix contains position index in shape (n_query, topk)
                'searchResDf': a paired search result dataframe
            )
        """
        if self.settings['qualified_token']==self.req_token or True:
            display('queryEmbedding shape is {}'.format(queryEmbedding.shape))
            searchRes = instanceRecommend.returnSimilarResultsOnSingleQuery(queryEmbedding, index=self.faissIndex, **retrievalKwargs)
            return searchRes
        else:
            abort(404)

    def ranking(self, queryEmbedding, docEmbeddings, rankingModel, pairedDF=None):
        display('needDocEmbeddings shape is {}'.format(docEmbeddings.shape))
        needQueryEmbeddings = np.resize(1*queryEmbedding, docEmbeddings.shape)
        display('needQueryEmbeddings shape is {}'.format(needQueryEmbeddings.shape))
        predLogits = rankingModel.predict(x=[needQueryEmbeddings, docEmbeddings], batch_size=self.settings['batch_size'])
        rankingRes = instanceGetdata.logitsToProbAndLabel(
            logits=predLogits, pairedDF=pairedDF #retrievalRes['searchResDf']
        )['comparedf']
        return rankingRes


api.add_resource(apiresource_recommend, '/search')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) #, use_reloader=False
    """better run below:
    #Linux: FLASK_APP=app.py FLASK_ENV=development flask run
    #Windows CMD: set FLASK_APP='app.py'; set FLASK_ENV='development'; flask run
    #Windows powershell:
    #$env:FLASK_APP='app.py'; $env:FLASK_ENV='development'; $env:FLASK_DEBUG=1; $env:FLASK_RUN_RELOAD=True; flask run
    #$env:FLASK_APP='app.py'; $env:FLASK_ENV='production'; $env:FLASK_DEBUG=0; $env:FLASK_RUN_RELOAD=False; flask run
    """