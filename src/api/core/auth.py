"""
Autenticación JWT sencilla (usuario/contraseña) para la API.

- Hash de contraseñas con `hashlib.pbkdf2_hmac` (stdlib, sin dependencias nuevas).
- Emisión/validación de JWT con `python-jose` (ya declarado en requirements-api.txt).
- Almacén de usuarios mínimo, sembrado desde la variable de entorno AUTH_USERS
  ("usuario:contraseña" separados por comas). Si no se define, se crea un
  usuario de desarrollo `doctor` / `patientia` (con aviso en logs).

Configuración por entorno:
- AUTH_SECRET_KEY: clave de firma del JWT (recomendado fijarla en producción).
- AUTH_USERS: "doctor:clave,admin:otra".
- AUTH_TOKEN_TTL_MIN: minutos de validez del token (por defecto 480).
"""
import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

ALGORITHM = "HS256"
_TOKEN_TTL_MIN = int(os.getenv("AUTH_TOKEN_TTL_MIN", "480"))

# Clave de firma. Si no se define, se genera una efímera (los tokens no
# sobreviven a reinicios). En producción, fijar AUTH_SECRET_KEY.
_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
if not _SECRET_KEY:
    _SECRET_KEY = secrets.token_urlsafe(48)
    logger.warning(
        "AUTH_SECRET_KEY no definida: usando una clave efímera generada. "
        "Define AUTH_SECRET_KEY para que los tokens persistan entre reinicios."
    )

_bearer = HTTPBearer(auto_error=False)


# ------------------------------ Password hashing ------------------------------
def hash_password(password: str, *, iterations: int = 200_000, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations, base64.b64encode(salt).decode(), base64.b64encode(dk).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# ------------------------------- User store ----------------------------------
def _load_users() -> Dict[str, str]:
    """Devuelve {username: password_hash} desde AUTH_USERS o un usuario dev."""
    raw = os.getenv("AUTH_USERS", "").strip()
    users: Dict[str, str] = {}
    if raw:
        for pair in raw.split(","):
            if ":" not in pair:
                continue
            username, password = pair.split(":", 1)
            username = username.strip()
            if username:
                users[username] = hash_password(password.strip())
    if not users:
        logger.warning("AUTH_USERS no definida: creando usuario de desarrollo 'doctor' / 'patientia'.")
        users["doctor"] = hash_password("patientia")
    return users


_USERS: Dict[str, str] = _load_users()


def authenticate_user(username: str, password: str) -> bool:
    stored = _USERS.get(username)
    if not stored:
        return False
    return verify_password(password, stored)


# --------------------------------- Tokens ------------------------------------
def create_access_token(username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=_TOKEN_TTL_MIN)).timestamp()),
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> Dict:
    return jwt.decode(token, _SECRET_KEY, algorithms=[ALGORITHM])


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """Dependencia FastAPI: valida el Bearer JWT y devuelve el username."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No autenticado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = _decode_token(credentials.credentials)
        username = payload.get("sub")
        if not username or username not in _USERS:
            raise JWTError("usuario no válido")
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o caducado",
            headers={"WWW-Authenticate": "Bearer"},
        )
