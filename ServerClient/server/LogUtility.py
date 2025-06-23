LOG_PATH = "robot.log"

def get_lines_containing_keyword(log_path, out_path, keywords: list[str]) -> list[str]:
    """
    Reads the log file and returns lines containing the specified keyword.
    """
    # create out file if it does not exist
    try:
        with open(out_path, encoding='utf-8', mode='a'):
            pass  # Just create the file
    except FileExistsError:
        pass  # File already exists, no action needed

    try:
        with open(LOG_PATH, encoding='utf-8', mode='r') as file:
            for line in file:
                if any(keyword in line for keyword in keywords):
                    with open(out_path, 'a') as out_file:
                        out_file.write(line)
    except FileNotFoundError:
        print(f"Log file {LOG_PATH} not found.")
    except Exception as e:
        print(f"An error occurred while reading the log file: {e}")

if __name__ == "__main__":
    get_lines_containing_keyword(LOG_PATH, "filtered_log.txt", ["WARNING  | PurePursuit:compute_drive_command():82"])