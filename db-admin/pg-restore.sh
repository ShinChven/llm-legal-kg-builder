#!/bin/bash

# ==============================================================================
# PostgreSQL Database Restore Script
#
# Description:
# This script automates the process of restoring a PostgreSQL database from a
# .dump file created with pg_dump. It interactively lists available backup
# files, prompts the user to select one, and asks for confirmation before
# dropping and recreating the target database.
#
# Prerequisites:
# 1.  `psql` and `pg_restore` command-line tools must be installed and in the PATH.
# 2.  A `.env` file must exist in the same directory as this script.
#
# .env file format:
# POSTGRES_USER="your_username"
# POSTGRES_PASSWORD="your_password"
# POSTGRES_DB="your_database_name"
# POSTGRES_HOST="localhost"
# POSTGRES_PORT="5432"
#
# ==============================================================================

# --- Configuration ---
# Directory where your .dump files are located.
DUMP_DIR="./dumps"

# --- Load variables from .env file ---
# Ensure .env file exists before proceeding.
if [ ! -f ./.env ]; then
    echo "Error: ./.env file not found! Please create it with your PostgreSQL credentials."
    exit 1
fi

# Source the .env file to load variables into the current environment.
# `set -a` exports all variables created, `set +a` stops exporting.
set -a
source ./.env
set +a
# --- End of .env loading ---

# --- Set PGPASSWORD for pg_restore and psql ---
# This prevents password prompts. The variable is unset at the end of the script.
export PGPASSWORD="${POSTGRES_PASSWORD}"

echo "--- PostgreSQL Restore Script ---"
echo "Database to restore to: ${POSTGRES_DB} on ${POSTGRES_HOST}:${POSTGRES_PORT} as user ${POSTGRES_USER}"
echo "Looking for backup files in: ${DUMP_DIR}"
echo "---------------------------------"

# --- Select backup file ---
BACKUP_FILES=()
i=0
echo "Available backup files:"
# Find all files ending in .dump in the specified directory.
for file in "${DUMP_DIR}"/*.dump; do
    # Check if the file actually exists to avoid errors if no .dump files are found.
    if [ -f "$file" ]; then
        BACKUP_FILES+=("$file")
        echo "  $((++i)). $(basename "$file")"
    fi
done

# Exit if no backup files were found.
if [ ${#BACKUP_FILES[@]} -eq 0 ]; then
    echo "No .dump backup files found in ${DUMP_DIR}. Exiting."
    unset PGPASSWORD
    exit 1
fi

# --- User selection loop ---
SELECTED_INDEX=-1
while true; do
    read -p "Enter the number of the backup file to restore (or 0 to exit): " CHOICE
    if [[ "$CHOICE" -eq 0 ]]; then
        echo "Restore cancelled."
        unset PGPASSWORD
        exit 0
    # Validate that the choice is a number within the valid range.
    elif [[ "$CHOICE" -ge 1 && "$CHOICE" -le ${#BACKUP_FILES[@]} ]]; then
        SELECTED_INDEX=$((CHOICE - 1))
        break
    else
        echo "Invalid choice. Please enter a number between 1 and ${#BACKUP_FILES[@]}, or 0 to exit."
    fi
done

SELECTED_BACKUP="${BACKUP_FILES[${SELECTED_INDEX}]}"
echo "You selected: $(basename "${SELECTED_BACKUP}")"
echo ""

# --- Confirmation for dropping and recreating database ---
while true; do
    read -p "WARNING: This will DROP and re-CREATE the '${POSTGRES_DB}' database. Are you sure? (y/N): " CONFIRM
    case "$CONFIRM" in
        [yY]|[yY][eE][sS])
            DROP_CREATE_DB=true
            break
            ;;
        [nN]|[nN][oO]|"") # Default to 'no' if user just presses Enter
            DROP_CREATE_DB=false
            break
            ;;
        *)
            echo "Invalid input. Please enter 'y' or 'n'."
            ;;
    esac
done

# --- Perform the restore ---
if [ "$DROP_CREATE_DB" = true ]; then
    echo "Dropping and re-creating database: ${POSTGRES_DB}..."
    # Connect to the 'postgres' database (or any other default db) to drop the target db.
    # You cannot be connected to a database to drop it.
    psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d postgres -c "DROP DATABASE IF EXISTS ${POSTGRES_DB};"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to drop database. Check permissions or if other users are connected."
        unset PGPASSWORD
        exit 1
    fi
    psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d postgres -c "CREATE DATABASE ${POSTGRES_DB};"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create database."
        unset PGPASSWORD
        exit 1
    fi
    echo "Database '${POSTGRES_DB}' dropped and re-created successfully."
fi

# Now restore the data using pg_restore
echo "Restoring data from ${SELECTED_BACKUP} into ${POSTGRES_DB}..."
pg_restore -h "${POSTGRES_HOST}" \
           -p "${POSTGRES_PORT}" \
           -U "${POSTGRES_USER}" \
           -d "${POSTGRES_DB}" \
           --no-owner \
           --clean \
           --if-exists \
           "${SELECTED_BACKUP}"

# Check the exit status of the last command (pg_restore)
if [ $? -eq 0 ]; then
    echo "Restore completed successfully!"
else
    echo "Restore failed! Check the output above for errors."
fi

# --- Clean up sensitive environment variables ---
unset PGPASSWORD
echo "Script finished."

