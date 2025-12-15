#!/bin/bash

DUMP_DIR="./dumps"

# --- Load variables from .env file ---
# Ensure .env file exists
if [ ! -f ./.env ]; then
    echo "Error: ./.env file not found! Please create it with your PostgreSQL credentials."
    exit 1
fi

# Source the .env file to load variables into the current environment
# Using 'set -a' makes all variables exported automatically
set -a
source ./.env
set +a
# --- End of .env loading ---

# --- Set PGPASSWORD for pg_dump ---
# pg_dump specifically looks for PGPASSWORD.
# We're taking it from your POSTGRES_PASSWORD variable loaded from .env
export PGPASSWORD="${POSTGRES_PASSWORD}"

# --- Create the backup directory if it doesn't exist ---
mkdir -p "${DUMP_DIR}"
if [ $? -ne 0 ]; then
    echo "Error: Could not create directory ${DUMP_DIR}. Check permissions."
    # Unset password before exiting if there's an error before pg_dump
    unset PGPASSWORD
    exit 1
fi
echo "Ensured backup directory exists: ${DUMP_DIR}"

# --- PostgreSQL Backup Command ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S) # Current time: 20250730_215811
BACKUP_FILE="${POSTGRES_DB}_backup_${TIMESTAMP}.dump"
FULL_BACKUP_PATH="${DUMP_DIR}/${BACKUP_FILE}"

echo "Starting backup of database: ${POSTGRES_DB} on ${POSTGRES_HOST}:${POSTGRES_PORT} as user ${POSTGRES_USER}"
echo "Saving backup to: ${FULL_BACKUP_PATH}"

pg_dump -h "${POSTGRES_HOST}" \
        -p "${POSTGRES_PORT}" \
        -U "${POSTGRES_USER}" \
        -Fc \
        -f "${FULL_BACKUP_PATH}" \
        "${POSTGRES_DB}"

# Check the exit status of pg_dump
if [ $? -eq 0 ]; then
    echo "Backup completed successfully to ${FULL_BACKUP_PATH}"
	# ls -lh the output folder
	ls -lh "${DUMP_DIR}"
else
    echo "Backup failed!"
fi

# --- Clean up sensitive environment variables ---
# It's good practice to unset PGPASSWORD immediately after the command
unset PGPASSWORD
