#include "linnet/package/fetch.hpp"
#include "linnet/support/text.hpp"

#include "test.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace linnet;

TEST("fetch: repositories are named as Hugging Face's cache names them") {
    const std::filesystem::path git = linnet_home() / "git";
    CHECK_EQ(repository_directory("https://github.com/owner/repo.git"),
             git / "github.com--owner--repo");
    CHECK_EQ(repository_directory("https://huggingface.co/org/model"),
             git / "huggingface.co--org--model");
    CHECK_EQ(repository_directory("git@github.com:owner/repo.git"),
             git / "github.com--owner--repo");
    CHECK_EQ(repository_directory("file:///tmp/a b/repo"), git / "file--tmp--a_b--repo");
    GitSource source;
    source.url = "https://github.com/owner/repo.git";
    source.subdir = "models/x";
    CHECK_EQ(package_directory(source, "c0ffee"),
             git / "github.com--owner--repo" / "snapshots" / "c0ffee" / "models" / "x");
}

TEST("fetch: a lock file reads back as written") {
    Lockfile lock;
    lock["https://github.com/a/b.git#tag=v1"] = std::string(40, 'a');
    lock["https://huggingface.co/c/d"] = std::string(40, 'b');
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "linnet-fetch-test.lock";
    {
        std::ofstream stream(path, std::ios::binary);
        stream << lockfile_text(lock);
    }
    const auto read = read_lockfile(path);
    CHECK(read.has_value());
    CHECK(*read == lock);
    {
        std::ofstream stream(path, std::ios::binary);
        stream << "version = 2\n";
    }
    CHECK(!read_lockfile(path).has_value());
    std::filesystem::remove(path);
    CHECK(read_lockfile(path).has_value()); // no lock yet: an empty one
}

TEST("fetch: a Windows command line splits back into the same words") {
    CHECK_EQ(command_line_word("plain"), std::string("plain"));
    CHECK_EQ(command_line_word(""), std::string("\"\""));
    CHECK_EQ(command_line_word("a b"), std::string("\"a b\""));
    CHECK_EQ(command_line_word(R"(say "hi")"), std::string(R"("say \"hi\"")"));
    // Backslashes double only before a quote, the closing one included.
    CHECK_EQ(command_line_word(R"(C:\dir with space\)"), std::string(R"("C:\dir with space\\")"));
}
