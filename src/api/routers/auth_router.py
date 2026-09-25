"""
Router de autenticación: login (usuario/contraseña -> JWT) y perfil actual.
"""
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from api.models.schemas import APIResponse
from api.core.auth import authenticate_user, create_access_token, get_current_user
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/auth/login", response_model=APIResponse)
async def login(request: Dict[str, Any]):
    """Autentica y devuelve un token JWT."""
    username = (request.get("username") or "").strip()
    password = request.get("password") or ""
    if not username or not password:
        raise HTTPException(status_code=400, detail="username y password son requeridos")

    if not authenticate_user(username, password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas",
        )

    token = create_access_token(username)
    return APIResponse(
        success=True,
        message="Autenticación correcta",
        data={"access_token": token, "token_type": "bearer", "username": username},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/auth/me", response_model=APIResponse)
async def me(current_user: str = Depends(get_current_user)):
    """Devuelve el usuario autenticado (valida el token)."""
    return APIResponse(
        success=True,
        message="Usuario autenticado",
        data={"username": current_user},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
