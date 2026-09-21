#include "linnet/diagnostic/json.hpp"

#include "test.hpp"

using namespace linnet;

TEST("json: strings are escaped and always valid Unicode") {
    CHECK_EQ(json_string("plain"), "\"plain\"");
    CHECK_EQ(json_string("a\"b\\c\n\t\r"), "\"a\\\"b\\\\c\\n\\t\\r\"");
    CHECK_EQ(json_string(std::string("\x01", 1)), "\"\\u0001\"");
    CHECK_EQ(json_string("\xC3\xA9"), "\"\xC3\xA9\"");
    CHECK_EQ(json_string("\xFF"), "\"\\ufffd\"");
}

TEST("json: diagnostics document") {
    SourceManager sources;
    const FileId file = sources.add_file("m.linnet", "let \xC3\xA9 = x\nnext").value();
    Diagnostic error;
    error.code = "E0001";
    error.message = "bad `x`";
    error.primary = {{file, 9, 10}, "here"};
    error.secondary.push_back({{file, 11, 15}, "related"});
    error.notes.push_back("a note");
    Diagnostic warning;
    warning.severity = Severity::Warning;
    warning.code = "W0001";
    warning.message = "global";
    const Diagnostic diagnostics[] = {error, warning};

    CHECK_EQ(
        render_json(sources, diagnostics),
        "{\"version\":1,\"diagnostics\":["
        "{\"code\":\"E0001\",\"severity\":\"error\",\"message\":\"bad `x`\",\"label\":\"here\","
        "\"location\":{\"file\":\"m.linnet\","
        "\"start\":{\"line\":1,\"column\":9,\"offset\":9},"
        "\"end\":{\"line\":1,\"column\":10,\"offset\":10}},"
        "\"related\":[{\"location\":{\"file\":\"m.linnet\","
        "\"start\":{\"line\":2,\"column\":1,\"offset\":11},"
        "\"end\":{\"line\":2,\"column\":5,\"offset\":15}},\"message\":\"related\"}],"
        "\"notes\":[\"a note\"],\"help\":[]},"
        "{\"code\":\"W0001\",\"severity\":\"warning\",\"message\":\"global\",\"label\":\"\","
        "\"location\":null,\"related\":[],\"notes\":[],\"help\":[]}],"
        "\"summary\":{\"errors\":1,\"warnings\":1}}\n");
}
