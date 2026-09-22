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
    CHECK_EQ(manifest->dependencies.at("foo").generic_string(), "pkg/../foo");
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
