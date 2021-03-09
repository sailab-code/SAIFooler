import os
import platform
import subprocess

executables_per_platform = {
    "Windows": "SAILenv.exe",
    "Linux": "SAILenv.run",
    "Darwin": os.path.join("SAILenv.app", "Contents", "MacOS", "SAILenv")
}

class SAILenvException(Exception):

    def __init__(self, stdout, stderr, *args: object) -> None:
        super().__init__(*args)

        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        return "STDOUT: \n" + \
            "\n".join(self.stdout) + "\n" + \
            "STDERR: \n" + \
            "\n".join(self.stderr)


class SAILenvManager:
    # process: Union[Popen[bytes], Popen[Any], None]

    def __init__(self, port="8085", prefix=None, sailenv_home="./"):
        """
        Inits the SAILenv Manager
        :param port: port on which to run sailenv
        :param prefix: optional prefix (can be used to set environment variables, such as DISPLAY in linux systems)
        :param sailenv_home: The path where the SAILenv executable is installed
        """
        curr_platform = platform.system()
        self.sailenv_exe = os.path.join(sailenv_home, executables_per_platform[curr_platform])
        self.port = port

        self.prefix = prefix
        self.process = None

    def start(self):
        command = []
        if self.prefix is not None:
            command.append(self.prefix)

        command.extend([self.sailenv_exe, "--port", self.port])
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # wait a second to be sure the executable has started
        try:
            self.process.wait(timeout=1)

            # if timeout is not expired, SAILenv has died without notice
            stdout = self.process.stderr.readlines()
            stderr = self.process.stderr.readlines()

            raise SAILenvException(stdout, stderr)

        except subprocess.TimeoutExpired:
            # if timeout expired, it means that SAILenv is still running
            return True

    def stop(self):
        self.process.kill()

    def restart(self):
        self.stop()
        self.start()