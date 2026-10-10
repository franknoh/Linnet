#include "linnet/package/manifest.hpp"

#include "test.hpp"

using namespace linnet;

TEST("manifest: minimal form with a path dependency") {
    const auto manifest =
        parse_manifest("[package]\nname = \"example-model\"\nversion = \"0.1.0\"\n"
                       "language = \"0.1\"\n\n[dependencies]\n"
                       "foo = { path = \"../foo\" }\n",
                       "pkg");
    CHECK(manifest.has_value());
    CHECK_EQ(manifest->name, "example-model");
    CHECK_EQ(manifest->dependencies.size(), 1U);
    CHECK_EQ(manifest->dependencies.at("foo").path.generic_string(), "pkg/../foo");
    CHECK(!manifest->dependencies.at("foo").git.has_value());
    CHECK(parse_manifest(default_manifest("model"), ".").has_value());
}

TEST("manifest: rejected forms name the problem") {
    const char* cases[] = {
        "",
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\n",
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\nlanguage = \"9.9\"\n",
        "[package]\nname = \"\"\nversion = \"0.1.0\"\nlanguage = \"0.1\"\n",
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\nlanguage = \"0.1\"\n[dependencies]\nstd = { "
        "path = \"x\" }\n",
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\nlanguage = \"0.1\"\n[dependencies]\nfoo = "
        "\"x\"\n",
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\nlanguage = \"0.1\"\n[dependencies]\nfoo = { "
        "git = \"x\" }\n",
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\nlanguage = \"0.1\"\n[dependencies]\nfoo = { "
        "path = \"x\", rev = \"y\" }\n",
        "[package\n",
    };
    for (const char* text : cases) {
        const auto manifest = parse_manifest(text, ".");
        CHECK(!manifest.has_value());
    }
    CHECK(is_valid_package_name("my_dep"));
    CHECK(!is_valid_package_name("my-dep"));
    CHECK(!is_valid_package_name("crate"));
    CHECK(!is_valid_package_name("1abc"));
}

TEST("manifest: git dependencies from GitHub, the Hub, or a URL") {
    const auto manifest = parse_manifest(
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\nlanguage = \"0.1\"\n[dependencies]\n"
        "layers = { github = \"owner/repo\", tag = \"v0.2\" }\n"
        "cards = { hf = \"org/model\", subdir = \"models/x\" }\n"
        "other = { git = \"https://example.com/a/b.git\", rev = \"4f2c\" }\n"
        "ssh = { git = \"git@github.com:owner/repo.git\", branch = \"main\" }\n",
        ".");
    CHECK(manifest.has_value());
    const auto source = [&](const char* name) {
        return manifest->dependencies.at(name).git.value_or(GitSource{});
    };
    const GitSource layers = source("layers");
    CHECK_EQ(layers.url, "https://github.com/owner/repo.git");
    CHECK_EQ(layers.key(), "https://github.com/owner/repo.git#tag=v0.2");
    const GitSource cards = source("cards");
    CHECK_EQ(cards.url, "https://huggingface.co/org/model");
    CHECK_EQ(cards.key(), "https://huggingface.co/org/model");
    CHECK_EQ(cards.subdir.generic_string(), "models/x");
    CHECK_EQ(source("other").reference, "rev=4f2c");
    CHECK(is_known_git_host(layers.url));
    CHECK(is_known_git_host(cards.url));
    CHECK(is_known_git_host("git@github.com:owner/repo.git"));
    CHECK(!is_known_git_host("https://example.com/a/b.git"));
    CHECK(!is_known_git_host("file:///tmp/repo"));
    CHECK_EQ(git_host("https://user@GitHub.com/a/b"), "github.com");
}

TEST("manifest: git dependencies that are refused") {
    const char* dependencies[] = {
        "x = { github = \"owner\" }",
        "x = { github = \"owner/repo/extra\" }",
        "x = { github = \"../repo\" }",
        "x = { hf = \"models/a/b\" }",
        "x = { git = \"ext::sh -c touch\" }",
        "x = { git = \"/local/path\" }",
        "x = { github = \"a/b\", tag = \"--upload-pack=x\" }",
        "x = { github = \"a/b\", tag = \"v1\", branch = \"main\" }",
        "x = { github = \"a/b\", subdir = \"../up\" }",
        "x = { github = \"a/b\", hf = \"a/b\" }",
        "x = { github = \"a/b\", depth = \"1\" }",
    };
    for (const char* dependency : dependencies) {
        const std::string text = std::string("[package]\nname = \"a\"\nversion = \"0.1.0\"\n"
                                             "language = \"0.1\"\n[dependencies]\n") +
                                 dependency + "\n";
        CHECK(!parse_manifest(text, ".").has_value());
    }
}
