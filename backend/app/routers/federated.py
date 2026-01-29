"""
Federated Learning Router
API endpoints for federated learning coordination
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import aiosqlite
from datetime import datetime
import sys
sys.path.append('..')

from app.ml.federated.server import create_federated_server
from app.ml.fusion.multimodal_fusion import create_multimodal_fusion

router = APIRouter()

# Global federated server (in production, use proper state management)
_federated_server = None


def get_federated_server():
    """Get or create federated server"""
    global _federated_server
    if _federated_server is None:
        global_model = create_multimodal_fusion()
        _federated_server = create_federated_server(global_model, num_clients=5)
    return _federated_server


class StartRound(BaseModel):
    num_clients: int = 5


@router.post("/rounds/start")
async def start_federated_round(request: StartRound):
    """Start a new federated learning round"""
    try:
        server = get_federated_server()
        round_info = server.start_round()
        
        # Save to database
        async with aiosqlite.connect("database/federated_learning.db") as db:
            cursor = await db.execute(
                """INSERT INTO federated_rounds (round_number, status, num_clients, started_at)
                   VALUES (?, ?, ?, ?)""",
                (round_info['round_number'], 'active', request.num_clients, 
                 round_info['started_at'])
            )
            round_id = cursor.lastrowid
            await db.commit()
        
        return {
            "round_id": round_id,
            "round_number": round_info['round_number'],
            "status": "active",
            "num_clients": request.num_clients,
            "message": "Federated learning round started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rounds/{round_id}/status")
async def get_round_status(round_id: int):
    """Get status of a federated learning round"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            
            # Get round info
            cursor = await db.execute(
                "SELECT * FROM federated_rounds WHERE id = ?",
                (round_id,)
            )
            round_info = await cursor.fetchone()
            
            if not round_info:
                raise HTTPException(status_code=404, detail="Round not found")
            
            # Get client updates
            cursor = await db.execute(
                "SELECT * FROM model_updates WHERE round_id = ?",
                (round_id,)
            )
            updates = await cursor.fetchall()
            
            return {
                "round": dict(round_info),
                "updates_received": len(updates),
                "updates": [dict(u) for u in updates]
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/rounds/{round_id}/metrics")
async def get_round_metrics(round_id: int):
    """Get aggregation metrics for a round"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM federated_rounds WHERE id = ?",
                (round_id,)
            )
            round_info = await cursor.fetchone()
            
            if not round_info:
                raise HTTPException(status_code=404, detail="Round not found")
            
            return dict(round_info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/model/weights")
async def get_global_model_weights():
    """Get current global model weights"""
    try:
        server = get_federated_server()
        weights = server.get_model_weights()
        
        # Convert to serializable format
        weights_info = {
            "num_parameters": sum(w.size for w in weights.values()),
            "parameter_names": list(weights.keys())
        }
        
        return weights_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_federated_stats():
    """Get overall federated learning statistics"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            
            # Total rounds
            cursor = await db.execute("SELECT COUNT(*) as count FROM federated_rounds")
            total_rounds = (await cursor.fetchone())['count']
            
            # Latest round
            cursor = await db.execute(
                "SELECT * FROM federated_rounds ORDER BY round_number DESC LIMIT 1"
            )
            latest_round = await cursor.fetchone()
            
            # Total updates
            cursor = await db.execute("SELECT COUNT(*) as count FROM model_updates")
            total_updates = (await cursor.fetchone())['count']
            
            return {
                "total_rounds": total_rounds,
                "latest_round": dict(latest_round) if latest_round else None,
                "total_updates": total_updates
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
