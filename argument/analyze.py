from typing import Optional

from pydantic import BaseModel, DirectoryPath


class AnalyzeArgument(BaseModel):
    log_dir: DirectoryPath
    select: Optional[list[DirectoryPath]]
