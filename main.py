from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from strawberry.fastapi import GraphQLRouter

from starlette.requests import Request

from auth.jwt import COOKIE_NAME, create_access_token, get_current_user, verify_password
from config import settings
from database import get_db
from gql_api.schema import schema
from models import User
from services.export import export_project
from services.storage import generate_presigned_upload, save_local_file


@asynccontextmanager
async def lifespan(app: FastAPI):
    from seed import init_database

    await init_database(reset=settings.db_reset_on_start)
    yield


app = FastAPI(title="AV Labeling Platform API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(settings.local_storage_dir)), name="files")


async def get_context(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        user = await get_current_user(request, db)
    except HTTPException:
        user = None
    return {"db": db, "user": user, "request": request}


graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    user: dict
    access_token: str


class PresignRequest(BaseModel):
    filename: str
    content_type: str


@app.post("/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user.id, user.role)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.jwt_expire_hours * 3600,
    )
    return LoginResponse(
        user={"id": str(user.id), "email": user.email, "name": user.name, "role": user.role.value},
        access_token=token,
    )


@app.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@app.get("/auth/me")
async def me(user: User = Depends(get_current_user)):
    return {"id": str(user.id), "email": user.email, "name": user.name, "role": user.role.value}


@app.post("/upload/presign")
async def presign(body: PresignRequest, user: User = Depends(get_current_user)):
    return generate_presigned_upload(body.filename, body.content_type)


@app.post("/upload/file")
async def upload_file(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    content = await file.read()
    result = await save_local_file(file.filename or "upload.jpg", content, file.content_type or "image/jpeg")
    return {
        "storage_key": result["storage_key"],
        "public_url": result["public_url"],
        "mime_type": result["mime_type"],
    }


@app.get("/export/{project_id}")
async def export(
    project_id: str,
    format: str = "json",
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from uuid import UUID

    try:
        content, filename, media_type = await export_project(db, UUID(project_id), format)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
