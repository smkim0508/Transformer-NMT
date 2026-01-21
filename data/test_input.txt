# Backend Pairing Service + API

Please read through this document to better understand how the backend service is set up, and where to find appropriate modules :\)

## Design Motivation & Notes

A large part of this codebase was written with scalability in mind, to easily be refactored into a more mature service that can serve lots of asynchronous/parallel requests without blocking. 

Currently, the MVP is designed with Flask, which is inherently synchronous, and the number of external APIs/dependencies are small (mainly PostgreSQL + SQLALchemy DB and Google GenAI Client). 

In the future, this should be refactored using an async server framework like FastAPI, which will integrate seamlessly with Pydantic models currently present (for auto validation of request params and response schemas). Moreover, the LLM client is initialized asynchronously, but a thin synchronous wrapper makes it compatible with the Flask version. Simply make inference via `llm_client.acreate()` instead of `llm_client.create_sync()`. 
- The database connection also needs to be converted to async sessions in the future, which is mostly handled under `/src/db/models/base/main_db.py` and the `app.py` set up.

Another feature that is designed with future scalability in mind is the wrapper repository and orchestrator classes (e.g. `PairingRepository` and `PairingOrchestrator`), which will help to cleanly port multiple dependencies during process duration (for a single request-scope) without messy function arguments. CRUD helpers also guide scalable codebase as we add more complex queries. 

Overall, the backend codebase is written modular, where necessary helpers and models can be found in intuitive directory names.

## Running Local Server
1. Setting up env: follow the example environment file `.env.example` and place the appropriate variables.

2. The main command to enter the WSGI application is `poetry run python wsgi.py`, when `cwd` is set as the `/src` directory.

## Understanding Request Lifecycle
1. In general, the entrypoint `wsgi.py` returns the main Flask main bound to `app.py`, and each request is routed via blueprints.

2. At each request scope, `@app.before_request` decorator will create instances of the LLM client and db session, managed by Flask's native global `g`, which lives only for the request duration.

3. Convenience repository and wrapper classes:
- The global `g` is wrapped inside `/src/api/dependencies.py`, cleanly returning `db_session` and `llm_client` instances to be used during process. 
- These variables should be wrapped in `PairingRepository` class that serves as the repository to hold `db_session` and `llm_client`, which helps to avoid passing around a (potentially) larger list of dependencies (you can simply reference them under `self`). *This will become more important as we grow the codebase, with additional external APIs.*
- Each time a request is made, this occurs right after initializing the dependencies in step 2, signaling the computation/query process to begin.

4. Work is done by helper functions. Any modular helper will come next, such as a db query & parser or an LLM client call. Usually, the helpers will return Pydantic models that hold the essential information *almost* ready to be passed to FE.

5. Pydantic model is dumped to be Flask-compatible. Since Flask natively doesn't support Pydantic models as response schema (unlike FastAPI), we need to dump the dict and wrap it with Flask's jsonifer: `flask.jsonify(X.model_dump())`. Return the jsonified value along with the appropriate HTTP status code.

## Backend Conventions
Below are some useful conventions when contributing to the backend service for code hygiene. **Please read this section before contributing to the codebase!**

1. Always return both a JSON response message and HTTP status code for all APIs.

2. Route naming convention should be something easily understood. There is no strict guideline we are following, but in general, the outer route (e.g. `student-dashboard/`) can be a descriptive noun, and the inner route (e.g. `view-events`) can be a verb + noun.

3. The main request entrypoint, defined under `/src/api/...` should be kept clean. This should mainly only initialize scope-local dependencies, kick-off processing with helper, minimally parse / format data, and return it. Longer, cluttering processes like LLM client call or db queries should be handled inside appropriately-defined helpers.

4. When creating new processes or helpers, find appropriate location by folder names, and contribute to existing files or create a new file. By convention, if a folder is vague and could hold more than one functionality, then create an appropriate lower-level sub-directory. 
    - Place `__init__.py` inside any importable directory if creating new locations.

5. When importing dependencies in files, start with the folder directory right after `/src`. For example, to import the `User` Pydantic model: `from common.types.user import User`. If dependencies do not resolve, make sure your `cwd` is set to `/src`. If issues persist, try adding `PYTHONPATH=.` before executing.

6. `/src/common` holds general-purpose Pydantic models (e.g. User, Event) that are central to our backend service, in addition to util functions. Please add any core models here.

7. To debug locally, use the `logger` defined under `/src/common/logging.py`. This helper can be used in 3 modes: `info`, `warning`, `error`. For a generic debugging session, do something like: `logger.info(f"This is a test logging statement.")` to print out the log statement to terminal. If possible, please try to remove clutters of local logs if the issue is resolved, but some logs can be kept if useful.

8. To work on a new feature, create an issue in `PeerPear Project`, then make a corresponding branch. Only close PR to main once feature is stable. Avoid making lots of incremental changes to main directly, unless it is absolutely non-breaking and minor. This may change in the future as we separate codebase into `prod` and `dev`.

## Dependency Management

Please use poetry to manage all backend dependencies. To add a new dependency, simply run `poetry add <dependency-group>`. If needed, refresh local dependencies by running `poetry install`.

### Dependencies for Heroku

Heroku uses a manually-exported `requirements.txt` file at root to manage dependencies. Whenever you add a new dependency to poetry, please update the `requirements.txt` file by executing `poetry export -f requirements.txt --without-hashes -o ../requirements.txt` in the project `/src` directory.