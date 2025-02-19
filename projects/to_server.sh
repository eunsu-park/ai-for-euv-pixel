#!/bin/bash

scp -P 10900 -r ./epic/* eunsu@100.10.3.191:/home/eunsu/Softwares/epic/
scp -P 10900 -r ./undine/* eunsu@100.10.3.191:/home/eunsu/Softwares/undine/


# # 서버 정보 설정
# SERVER_USER="eunsu"    # 서버 사용자명
# SERVER_HOST="100.10.3.191:"  # 서버 주소
# SERVER_PORT=10900
# REMOTE_PATH="/home/eunsu/Softwares/" # 서버에서의 저장 경로

# # 업로드할 폴더 목록
# PROJECT_EPIC="epic"
# PROJECT_UNDINE="undine"

# # 업로드할 목록 초기화
# UPLOAD_EPIC=false
# UPLOAD_UNDINE=false

# # 옵션 처리
# while getopts "euh" opt; do
#   case ${opt} in
#     e ) UPLOAD_EPIC=true ;;
#     u ) UPLOAD_UNDINE=true ;;
#     h ) 
#         echo "사용법: $0 [-e] [-u] [-h]"
#         echo "  -e : epic만 업로드"
#         echo "  -u : undine만 업로드"
#         echo "  -h : 도움말 출력"
#         exit 0
#         ;;
#     \? )
#         echo "잘못된 옵션입니다. -h 옵션으로 사용법을 확인하세요."
#         exit 1
#         ;;
#   esac
# done

# # 옵션이 없을 경우 모든 폴더 업로드
# if ! $UPLOAD_EPIC && ! $UPLOAD_UNDINE; then
#   UPLOAD_EPIC=true
#   UPLOAD_UNDINE=true
# fi

# # SCP 실행
# if $UPLOAD_EPIC; then
#   echo "Uploading $PROJECT_EPIC to $SERVER_USER@$SERVER_HOST:$REMOTE_PATH"
#   scp -P "$SERVER_PORT" -r "$PROJECT_EPIC/*" "$SERVER_USER@$SERVER_HOST:$REMOTE_PATH/$PROJECT_EPIC/"
# fi

# if $UPLOAD_UNDINE; then
#   echo "Uploading $PROJECT_UNDINE to $SERVER_USER@$SERVER_HOST:$REMOTE_PATH"
#   scp -P "$SERVER_PORT" -r "$PROJECT_UNDINE/*" "$SERVER_USER@$SERVER_HOST:$REMOTE_PATH/$PROJECT_UNDINE/"
# fi

# echo "업로드 완료!"

