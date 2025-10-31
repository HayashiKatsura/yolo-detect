import os
import mimetypes

MIME_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.zip': 'application/zip',
    '.rar': 'application/x-rar-compressed',
    '.7z': 'application/x-7z-compressed',
    '.pt': 'application/octet-stream', 
    '.pth': 'application/octet-stream',  
    '.pkl': 'application/octet-stream', 
    '.yaml': 'text/yaml',
    '.yml': 'text/yaml',
    '.txt': 'text/plain',
    '.log': 'text/plain',
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.mp4': 'video/mp4',
}



def get_mime_type(filename: str) -> str:
    """根据扩展名获取 MIME 类型，优先查自定义表"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in MIME_TYPES:
        return MIME_TYPES[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"