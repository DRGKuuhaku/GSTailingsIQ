"""
TailingsIQ - Document Management API Endpoints

This module provides FastAPI endpoints for document management operations
including upload, search, analysis, and retrieval of TSF-related documents.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import uuid
import mimetypes
import io
from pathlib import Path

from ..core.database import get_db
from ..core.security import get_current_user, require_permission
from ..models.user import User
from ..models.document import (
    Document, DocumentCreate, DocumentUpdate, DocumentResponse, 
    DocumentSearchQuery, DocumentAnalysisResult, DocumentType, DocumentStatus
)
from ..services.document_intelligence import DocumentIntelligenceService
from ..services.rag_processor import RAGProcessor
from ..utils.helpers import generate_uuid, validate_file_type, get_file_hash

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
doc_intelligence = DocumentIntelligenceService()
rag_processor = RAGProcessor()

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    document_type: DocumentType = Form(DocumentType.TECHNICAL_REPORT),
    facility_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a new document to the repository.

    Args:
        file: The file to upload
        title: Optional document title (extracted from file if not provided)
        description: Optional document description
        document_type: Type of document being uploaded
        facility_id: Associated facility ID
        tags: Comma-separated tags
        current_user: Current authenticated user
        db: Database session

    Returns:
        DocumentResponse: Created document information
    """
    try:
        # Validate file type and size
        if not validate_file_type(file.content_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file.content_type} is not supported"
            )

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        # Validate file size (from settings)
        from ..core.config import settings
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )

        # Generate file hash for deduplication
        file_hash = get_file_hash(file_content)

        # Check for duplicate files
        existing_doc = db.query(Document).filter(Document.file_hash == file_hash).first()
        if existing_doc:
            logger.info(f"Duplicate file detected: {file.filename}")
            return DocumentResponse.from_orm(existing_doc)

        # Generate unique filename
        file_extension = Path(file.filename).suffix
        unique_filename = f"{generate_uuid()}{file_extension}"

        # Save file to storage
        file_path = await save_file_to_storage(file_content, unique_filename)

        # Extract metadata from file
        metadata = await doc_intelligence.extract_metadata(file_content, file.content_type)

        # Create document record
        document_data = DocumentCreate(
            filename=file.filename,
            file_path=str(file_path),
            file_size=file_size,
            content_type=file.content_type,
            file_hash=file_hash,
            title=title or metadata.get("title", file.filename),
            description=description or metadata.get("description"),
            document_type=document_type,
            facility_id=facility_id,
            tags=tags.split(",") if tags else [],
            metadata=metadata,
            uploaded_by=current_user.id
        )

        # Create document in database
        db_document = Document(**document_data.dict())
        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            document_id=db_document.id,
            file_content=file_content,
            content_type=file.content_type
        )

        logger.info(f"Document uploaded successfully: {db_document.id}")
        return DocumentResponse.from_orm(db_document)

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    document_type: Optional[DocumentType] = None,
    facility_id: Optional[str] = None,
    search: Optional[str] = None,
    tags: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List documents with filtering and pagination.

    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        document_type: Filter by document type
        facility_id: Filter by facility ID
        search: Search term for title/description
        tags: Comma-separated tags to filter by
        date_from: Filter documents from this date
        date_to: Filter documents to this date
        current_user: Current authenticated user
        db: Database session

    Returns:
        List[DocumentResponse]: List of documents
    """
    try:
        query = db.query(Document)

        # Apply filters
        if document_type:
            query = query.filter(Document.document_type == document_type.value)

        if facility_id:
            query = query.filter(Document.facility_id == facility_id)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (Document.title.ilike(search_term)) |
                (Document.description.ilike(search_term))
            )

        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            query = query.filter(Document.tags.op("&&")(tag_list))

        if date_from:
            query = query.filter(Document.created_at >= date_from)

        if date_to:
            query = query.filter(Document.created_at <= date_to)

        # Apply pagination and ordering
        documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()

        return [DocumentResponse.from_orm(doc) for doc in documents]

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific document by ID.

    Args:
        document_id: Document ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        DocumentResponse: Document information
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Update access count and last accessed
        document.access_count = (document.access_count or 0) + 1
        document.last_accessed = datetime.utcnow()
        db.commit()

        return DocumentResponse.from_orm(document)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}"
        )

@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Download a document file.

    Args:
        document_id: Document ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        FileResponse: Document file
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Check if file exists
        file_path = Path(document.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found"
            )

        # Update download count
        document.download_count = (document.download_count or 0) + 1
        db.commit()

        return FileResponse(
            path=str(file_path),
            filename=document.filename,
            media_type=document.content_type
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download document: {str(e)}"
        )

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    document_update: DocumentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update document metadata.

    Args:
        document_id: Document ID
        document_update: Updated document information
        current_user: Current authenticated user
        db: Database session

    Returns:
        DocumentResponse: Updated document information
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Check permissions (only uploader or admin can update)
        if document.uploaded_by != current_user.id and not require_permission(current_user, "can_manage_documents"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this document"
            )

        # Update fields
        update_data = document_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(document, field, value)

        document.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(document)

        logger.info(f"Document updated: {document_id}")
        return DocumentResponse.from_orm(document)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}"
        )

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a document.

    Args:
        document_id: Document ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        dict: Success message
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Check permissions (only uploader or admin can delete)
        if document.uploaded_by != current_user.id and not require_permission(current_user, "can_manage_documents"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this document"
            )

        # Delete file from storage
        try:
            file_path = Path(document.file_path)
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete file {document.file_path}: {e}")

        # Delete from database
        db.delete(document)
        db.commit()

        logger.info(f"Document deleted: {document_id}")
        return {"message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.post("/search", response_model=List[DocumentResponse])
async def search_documents(
    search_query: DocumentSearchQuery,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Advanced document search with semantic capabilities.

    Args:
        search_query: Search parameters
        current_user: Current authenticated user
        db: Database session

    Returns:
        List[DocumentResponse]: Search results
    """
    try:
        # Use RAG processor for semantic search
        search_results = await rag_processor.search_documents(
            query=search_query.query,
            filters=search_query.filters,
            limit=search_query.limit
        )

        # Get document IDs from search results
        document_ids = [result.get("document_id") for result in search_results if result.get("document_id")]

        # Fetch documents from database
        documents = db.query(Document).filter(Document.id.in_(document_ids)).all()

        # Sort by search relevance
        document_dict = {doc.id: doc for doc in documents}
        sorted_documents = [document_dict[doc_id] for doc_id in document_ids if doc_id in document_dict]

        return [DocumentResponse.from_orm(doc) for doc in sorted_documents]

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document search failed: {str(e)}"
        )

@router.post("/{document_id}/analyze", response_model=DocumentAnalysisResult)
async def analyze_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze document content with AI.

    Args:
        document_id: Document ID
        background_tasks: Background task queue
        current_user: Current authenticated user
        db: Database session

    Returns:
        DocumentAnalysisResult: Analysis results
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Check if analysis already exists and is recent
        if (document.analysis_result and 
            document.analysis_completed_at and 
            document.analysis_completed_at > datetime.utcnow() - timedelta(days=7)):
            return DocumentAnalysisResult(**document.analysis_result)

        # Schedule background analysis
        background_tasks.add_task(
            analyze_document_background,
            document_id=document_id,
            user_id=current_user.id
        )

        return DocumentAnalysisResult(
            document_id=document_id,
            status="processing",
            message="Document analysis started"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting document analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze document: {str(e)}"
        )

@router.get("/{document_id}/analysis", response_model=DocumentAnalysisResult)
async def get_document_analysis(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get document analysis results.

    Args:
        document_id: Document ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        DocumentAnalysisResult: Analysis results
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        if not document.analysis_result:
            return DocumentAnalysisResult(
                document_id=document_id,
                status="not_analyzed",
                message="Document has not been analyzed yet"
            )

        return DocumentAnalysisResult(**document.analysis_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analysis: {str(e)}"
        )

# Helper functions

async def save_file_to_storage(file_content: bytes, filename: str) -> Path:
    """Save file to storage and return path."""
    from ..core.config import settings

    storage_path = Path(settings.UPLOAD_DIR)
    storage_path.mkdir(parents=True, exist_ok=True)

    file_path = storage_path / filename

    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path

async def process_document_background(document_id: str, file_content: bytes, content_type: str):
    """Background task to process uploaded document."""
    try:
        # Extract text content
        extracted_text = await doc_intelligence.extract_text(file_content, content_type)

        # Generate embeddings and add to vector database
        await rag_processor.add_document(
            document_id=document_id,
            content=extracted_text,
            metadata={"content_type": content_type}
        )

        # Update document with processing results
        from ..core.database import SessionLocal
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processed_at = datetime.utcnow()
                document.processing_status = "completed"
                document.extracted_text = extracted_text
                db.commit()
        finally:
            db.close()

        logger.info(f"Document processing completed: {document_id}")

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")

        # Update document with error status
        from ..core.database import SessionLocal
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processing_status = "failed"
                document.processing_error = str(e)
                db.commit()
        finally:
            db.close()

async def analyze_document_background(document_id: str, user_id: int):
    """Background task to analyze document with AI."""
    try:
        from ..core.database import SessionLocal
        db = SessionLocal()

        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return

            # Perform AI analysis
            analysis_result = await doc_intelligence.analyze_document(
                document_id=document_id,
                content=document.extracted_text,
                metadata=document.metadata
            )

            # Update document with analysis results
            document.analysis_result = analysis_result
            document.analysis_completed_at = datetime.utcnow()
            document.analysis_status = "completed"
            db.commit()

            logger.info(f"Document analysis completed: {document_id}")

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error analyzing document {document_id}: {str(e)}")

        # Update document with error status
        from ..core.database import SessionLocal
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.analysis_status = "failed"
                document.analysis_error = str(e)
                db.commit()
        finally:
            db.close()