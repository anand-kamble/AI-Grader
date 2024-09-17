### With Ollama/llama3.1

Setting Used:
```
local_llm_config = {
    "model_list": [
        {
            "model_name": "ollama/llama3.1",
            "litellm_params": {
                "model": "ollama/llama3.1",
                "api_base": "http://localhost:11434",
            },
        }
    ]
}

answer: AnswerResponse = ask(
    query="What manufacturing challenges are unique to bispecific antibodies?",
    settings=Settings(
        llm="ollama/llama3.1",
        llm_config=local_llm_config,
        summary_llm="ollama/llama3.1",
        summary_llm_config=local_llm_config,
        paper_directory="solutions_questions",
    ),
)
```

Output:
```
[09:04:20] Starting paper search for 'Here are three unique keyword  
           searches with specified year ranges that can help answer  
           your question:'.                                          
           New file to index:                                        
           Introduction_to_data_mining_2020_tan_solution_manual.pdf..
           .                                                         
           New file to index: Question Bank 1 (Tan et al 2nd         
           Edition).txt...                                           
           New file to index: Introducation to data mining,          
           solutions, 1st ed_book.pdf...                             
[09:04:21] New file to index: Clustering.pdf...                      
           New file to index: Question Bank 1 (Tan et al 2nd         
           Edition).pdf...                                           
[09:04:25] SEMANTIC_SCHOLAR_API_KEY environment variable not set.    
           Semantic Scholar API rate limits may apply.               
           CROSSREF_MAILTO environment variable not set. Crossref API
           rate limits may apply.                                    
           CROSSREF_API_KEY environment variable not set. Crossref   
           API rate limits may apply.                                
[09:04:26] CROSSREF_API_KEY environment variable not set. Crossref   
           API rate limits may apply.                                
           Metadata not found for Introduction to Data Mining in     
           CrossrefProvider.                                         

Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
LiteLLM.Info: If you need to debug this error, use `litellm.set_verbose=True'.

/home/amk23j/.conda/envs/playground/lib/python3.12/asyncio/base_events.py:1986: RuntimeWarning: coroutine 'OpenAIChatCompletion.aembedding' was never awaited
  handle = None  # Needed to break cycles when an exception occurs.
RuntimeWarning: Enable tracemalloc to get the object allocation traceback
Failed to execute tool call for tool paper_search.
  + Exception Group Traceback (most recent call last):
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/aviary/env.py", line 197, in _exec_tool_call
  |     content = await tool._tool_fn(
  |               ^^^^^^^^^^^^^^^^^^^^
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/tools.py", line 127, in paper_search
  |     index = await get_directory_index(settings=self.settings)
  |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/search.py", line 481, in get_directory_index
  |     async with anyio.create_task_group() as tg:
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 680, in __aexit__
  |     raise BaseExceptionGroup(
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1238, in aembedding
    |     headers, response = await self.make_openai_embedding_request(
    |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1192, in make_openai_embedding_request
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1185, in make_openai_embedding_request
    |     raw_response = await openai_aclient.embeddings.with_raw_response.create(
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_legacy_response.py", line 367, in wrapped
    |     return cast(LegacyAPIResponse[R], await func(*args, **kwargs))
    |                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/resources/embeddings.py", line 237, in create
    |     return await self._post(
    |            ^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1816, in post
    |     return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1510, in request
    |     return await self._request(
    |            ^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1611, in _request
    |     raise self._make_status_error_from_response(err.response) from None
    | openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    | 
    | During handling of the above exception, another exception occurred:
    | 
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/main.py", line 3200, in aembedding
    |     response = await init_response
    |                ^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1275, in aembedding
    |     raise OpenAIError(
    | litellm.llms.OpenAI.openai.OpenAIError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    | 
    | During handling of the above exception, another exception occurred:
    | 
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/search.py", line 377, in process_file
    |     await tmp_docs.aadd(
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/docs.py", line 369, in aadd
    |     if await self.aadd_texts(texts, doc, all_settings, embedding_model):
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/docs.py", line 414, in aadd_texts
    |     await embedding_model.embed_documents(texts=[t.text for t in texts])
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/llms.py", line 74, in embed_documents
    |     response = await aembedding(
    |                ^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 1595, in wrapper_async
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 1415, in wrapper_async
    |     result = await original_function(*args, **kwargs)
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/main.py", line 3209, in aembedding
    |     raise exception_type(
    |           ^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 8196, in exception_type
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 6531, in exception_type
    |     raise AuthenticationError(
    | litellm.exceptions.AuthenticationError: litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    +------------------------------------
[09:04:27] Starting paper search for 'Bispecific antibody            
           manufacturing challenges, 2018-2024'.                     
           New file to index: Question Bank 1 (Tan et al 2nd         
           Edition).txt...                                           
           New file to index:                                        
           Introduction_to_data_mining_2020_tan_solution_manual.pdf..
           .                                                         
           New file to index: Clustering.pdf...                      
           New file to index: Question Bank 1 (Tan et al 2nd         
           Edition).pdf...                                           
[09:04:28] New file to index: Introducation to data mining,          
           solutions, 1st ed_book.pdf...                             

Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
LiteLLM.Info: If you need to debug this error, use `litellm.set_verbose=True'.

Failed to execute tool call for tool paper_search.
  + Exception Group Traceback (most recent call last):
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/aviary/env.py", line 197, in _exec_tool_call
  |     content = await tool._tool_fn(
  |               ^^^^^^^^^^^^^^^^^^^^
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/tools.py", line 127, in paper_search
  |     index = await get_directory_index(settings=self.settings)
  |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/search.py", line 481, in get_directory_index
  |     async with anyio.create_task_group() as tg:
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 680, in __aexit__
  |     raise BaseExceptionGroup(
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1238, in aembedding
    |     headers, response = await self.make_openai_embedding_request(
    |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1192, in make_openai_embedding_request
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1185, in make_openai_embedding_request
    |     raw_response = await openai_aclient.embeddings.with_raw_response.create(
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_legacy_response.py", line 367, in wrapped
    |     return cast(LegacyAPIResponse[R], await func(*args, **kwargs))
    |                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/resources/embeddings.py", line 237, in create
    |     return await self._post(
    |            ^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1816, in post
    |     return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1510, in request
    |     return await self._request(
    |            ^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1611, in _request
    |     raise self._make_status_error_from_response(err.response) from None
    | openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    | 
    | During handling of the above exception, another exception occurred:
    | 
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/main.py", line 3200, in aembedding
    |     response = await init_response
    |                ^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1275, in aembedding
    |     raise OpenAIError(
    | litellm.llms.OpenAI.openai.OpenAIError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    | 
    | During handling of the above exception, another exception occurred:
    | 
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/search.py", line 377, in process_file
    |     await tmp_docs.aadd(
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/docs.py", line 369, in aadd
    |     if await self.aadd_texts(texts, doc, all_settings, embedding_model):
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/docs.py", line 414, in aadd_texts
    |     await embedding_model.embed_documents(texts=[t.text for t in texts])
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/llms.py", line 74, in embed_documents
    |     response = await aembedding(
    |                ^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 1595, in wrapper_async
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 1415, in wrapper_async
    |     result = await original_function(*args, **kwargs)
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/main.py", line 3209, in aembedding
    |     raise exception_type(
    |           ^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 8196, in exception_type
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 6531, in exception_type
    |     raise AuthenticationError(
    | litellm.exceptions.AuthenticationError: litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    +------------------------------------
[09:04:33] Starting paper search for 'Antibody production            
           difficulties in bispecific format, 2015-'.                
           New file to index: Question Bank 1 (Tan et al 2nd         
           Edition).txt...                                           
           New file to index: Question Bank 1 (Tan et al 2nd         
           Edition).pdf...                                           
           New file to index: Introducation to data mining,          
           solutions, 1st ed_book.pdf...                             
           New file to index:                                        
           Introduction_to_data_mining_2020_tan_solution_manual.pdf..
           .                                                         
           New file to index: Clustering.pdf...                      

Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
LiteLLM.Info: If you need to debug this error, use `litellm.set_verbose=True'.

Failed to execute tool call for tool paper_search.
  + Exception Group Traceback (most recent call last):
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/aviary/env.py", line 197, in _exec_tool_call
  |     content = await tool._tool_fn(
  |               ^^^^^^^^^^^^^^^^^^^^
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/tools.py", line 127, in paper_search
  |     index = await get_directory_index(settings=self.settings)
  |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/search.py", line 481, in get_directory_index
  |     async with anyio.create_task_group() as tg:
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 680, in __aexit__
  |     raise BaseExceptionGroup(
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1238, in aembedding
    |     headers, response = await self.make_openai_embedding_request(
    |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1192, in make_openai_embedding_request
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1185, in make_openai_embedding_request
    |     raw_response = await openai_aclient.embeddings.with_raw_response.create(
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_legacy_response.py", line 367, in wrapped
    |     return cast(LegacyAPIResponse[R], await func(*args, **kwargs))
    |                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/resources/embeddings.py", line 237, in create
    |     return await self._post(
    |            ^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1816, in post
    |     return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1510, in request
    |     return await self._request(
    |            ^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1611, in _request
    |     raise self._make_status_error_from_response(err.response) from None
    | openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    | 
    | During handling of the above exception, another exception occurred:
    | 
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/main.py", line 3200, in aembedding
    |     response = await init_response
    |                ^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1275, in aembedding
    |     raise OpenAIError(
    | litellm.llms.OpenAI.openai.OpenAIError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    | 
    | During handling of the above exception, another exception occurred:
    | 
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/search.py", line 377, in process_file
    |     await tmp_docs.aadd(
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/docs.py", line 369, in aadd
    |     if await self.aadd_texts(texts, doc, all_settings, embedding_model):
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/docs.py", line 414, in aadd_texts
    |     await embedding_model.embed_documents(texts=[t.text for t in texts])
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/llms.py", line 74, in embed_documents
    |     response = await aembedding(
    |                ^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 1595, in wrapper_async
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 1415, in wrapper_async
    |     result = await original_function(*args, **kwargs)
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/main.py", line 3209, in aembedding
    |     raise exception_type(
    |           ^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 8196, in exception_type
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 6531, in exception_type
    |     raise AuthenticationError(
    | litellm.exceptions.AuthenticationError: litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    +------------------------------------
[09:04:38] Starting paper search for 'Scale-up issues for dual       
           targeting antibodies, 2012-2020'.                         
           New file to index: Question Bank 1 (Tan et al 2nd         
           Edition).txt...                                           
           New file to index: Question Bank 1 (Tan et al 2nd         
           Edition).pdf...                                           
[09:04:39] New file to index:                                        
           Introduction_to_data_mining_2020_tan_solution_manual.pdf..
           .                                                         
           New file to index: Clustering.pdf...                      
           New file to index: Introducation to data mining,          
           solutions, 1st ed_book.pdf...                             

Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
LiteLLM.Info: If you need to debug this error, use `litellm.set_verbose=True'.

Failed to execute tool call for tool paper_search.
  + Exception Group Traceback (most recent call last):
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/aviary/env.py", line 197, in _exec_tool_call
  |     content = await tool._tool_fn(
  |               ^^^^^^^^^^^^^^^^^^^^
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/tools.py", line 127, in paper_search
  |     index = await get_directory_index(settings=self.settings)
  |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/search.py", line 481, in get_directory_index
  |     async with anyio.create_task_group() as tg:
  |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/anyio/_backends/_asyncio.py", line 680, in __aexit__
  |     raise BaseExceptionGroup(
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1238, in aembedding
    |     headers, response = await self.make_openai_embedding_request(
    |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1192, in make_openai_embedding_request
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1185, in make_openai_embedding_request
    |     raw_response = await openai_aclient.embeddings.with_raw_response.create(
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_legacy_response.py", line 367, in wrapped
    |     return cast(LegacyAPIResponse[R], await func(*args, **kwargs))
    |                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/resources/embeddings.py", line 237, in create
    |     return await self._post(
    |            ^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1816, in post
    |     return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1510, in request
    |     return await self._request(
    |            ^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/openai/_base_client.py", line 1611, in _request
    |     raise self._make_status_error_from_response(err.response) from None
    | openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    | 
    | During handling of the above exception, another exception occurred:
    | 
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/main.py", line 3200, in aembedding
    |     response = await init_response
    |                ^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/llms/OpenAI/openai.py", line 1275, in aembedding
    |     raise OpenAIError(
    | litellm.llms.OpenAI.openai.OpenAIError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    | 
    | During handling of the above exception, another exception occurred:
    | 
    | Traceback (most recent call last):
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/search.py", line 377, in process_file
    |     await tmp_docs.aadd(
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/docs.py", line 369, in aadd
    |     if await self.aadd_texts(texts, doc, all_settings, embedding_model):
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/docs.py", line 414, in aadd_texts
    |     await embedding_model.embed_documents(texts=[t.text for t in texts])
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/llms.py", line 74, in embed_documents
    |     response = await aembedding(
    |                ^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 1595, in wrapper_async
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 1415, in wrapper_async
    |     result = await original_function(*args, **kwargs)
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/main.py", line 3209, in aembedding
    |     raise exception_type(
    |           ^^^^^^^^^^^^^^^
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 8196, in exception_type
    |     raise e
    |   File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/litellm/utils.py", line 6531, in exception_type
    |     raise AuthenticationError(
    | litellm.exceptions.AuthenticationError: litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-12345***********************cdef. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
    +------------------------------------
Failed to execute tool call for tool gather_evidence.
Traceback (most recent call last):
  File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/aviary/env.py", line 197, in _exec_tool_call
    content = await tool._tool_fn(
              ^^^^^^^^^^^^^^^^^^^^
  File "/home/amk23j/.cache/pypoetry/virtualenvs/auto-grader-poN1QxeZ-py3.12/lib/python3.12/site-packages/paperqa/agents/tools.py", line 188, in gather_evidence
    raise EmptyDocsError("Not gathering evidence due to having no papers.")
paperqa.agents.tools.EmptyDocsError: Not gathering evidence due to having no papers.
[09:04:45] Generating answer for 'What manufacturing challenges are  
           unique to bispecific antibodies?'.                        
           Status: Paper Count=0 | Relevant Papers=0 | Current       V
           Evidence=0 | Current Cost=$0.0000                         
           Answer: Unfortunately, I cannot provide an answer to this 
           question as there is insufficient information provided in 
           the context regarding manufacturing challenges specific to
           bispecific antibodies.
```